use glib;
use glib::subclass;
use glib::subclass::prelude::*;

use gst;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base;
use gst_base::subclass::prelude::*;
use gst_video;

use std::result::Result;
use std::i32;
use std::sync::Mutex;

use opencv::{
    objdetect,
    prelude::*,
    core::{
        Rect,
        Scalar
    },
    imgproc,
    types::VectorOfPoint,
};

use std::mem::transmute;


#[cfg_attr(feature = "examples_system_alloc", global_allocator)]
#[cfg(feature = "examples_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

// Default values of properties
const DEFAULT_INVERT: bool = false;
const DEFAULT_SHIFT: u32 = 0;

// Property value storage
#[derive(Debug, Clone, Copy)]
struct Settings {
    invert: bool,
    shift: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            invert: DEFAULT_INVERT,
            shift: DEFAULT_SHIFT,
        }
    }
}

// Metadata for the properties
static PROPERTIES: [subclass::Property; 2] = [
    subclass::Property("invert", |name| {
        glib::ParamSpec::boolean(
            name,
            "Invert",
            "Invert grayscale output",
            DEFAULT_INVERT,
            glib::ParamFlags::READWRITE,
        )
    }),
    subclass::Property("shift", |name| {
        glib::ParamSpec::uint(
            name,
            "Shift",
            "Shift grayscale output (wrapping around)",
            0,
            255,
            DEFAULT_SHIFT,
            glib::ParamFlags::READWRITE,
        )
    }),
];

// Stream-specific state, i.e. video format configuration
struct State {
    in_info: gst_video::VideoInfo,
    out_info: gst_video::VideoInfo,
}

// Struct containing all the element data
struct Segmentation {
    cat: gst::DebugCategory,
    settings: Mutex<Settings>,
    state: Mutex<Option<State>>
}

impl Segmentation {
    // fn DrawLines(Img: &mut Mat, BBox: &Mat) {
        // let nrows: i32 = BBox.rows().unwrap();
        // let lines_color = core::Scalar::new(255.0, 0.0, 0.0, 1.0);

        // for index in 0..nrows {
        //     let p1_a: f32 = BBox.at_2d<float>(index, 0).unwrap();
            
        //     let p1 = Point2i::new(p1_a, BBox.at_2d(index, 1));
        //     let p2 = Point2i::new(BBox.at_2d((index + 1) % nrows, 0), BBox.at_2d((index + 1) % nrows, 1));

        //     imgproc::line(Img, p1, p2, lines_color, 3, 8, 0);
        // }

    // }
}

// This trait registers our type with the GObject object system and
// provides the entry points for creating a new instance and setting
// up the class data
impl ObjectSubclass for Segmentation {
    const NAME: &'static str = "Segmentation";
    type ParentType = gst_base::BaseTransform;
    type Instance = gst::subclass::ElementInstanceStruct<Self>;
    type Class = subclass::simple::ClassStruct<Self>;

    // This macro provides some boilerplate
    glib_object_subclass!();

    // Called when a new instance is to be created. We need to return an instance
    // of our struct here.
    fn new() -> Self {
        Self {
            cat: gst::DebugCategory::new(
                "tf_segmentation",
                gst::DebugColorFlags::empty(),
                Some("video to depth inference"),
            ),
            settings: Mutex::new(Default::default()),
            state: Mutex::new(None)
        }
    }

    // Called exactly once when registering the type. Used for
    // setting up metadata for all instances, e.g. the name and
    // classification and the pad templates with their caps.
    //
    // Actual instances can create pads based on those pad templates
    // with a subset of the caps given here. In case of basetransform,
    // a "src" and "sink" pad template are required here and the base class
    // will automatically instantiate pads for them.
    //
    // Our element here can convert Rgb to Rgb or Gray8, both being grayscale.
    fn class_init(klass: &mut subclass::simple::ClassStruct<Self>) {
        // Set the element specific metadata. This information is what
        // is visible from gst-inspect-1.0 and can also be programatically
        // retrieved from the gst::Registry after initial registration
        // without having to load the plugin in memory.
        klass.set_metadata(
            "Segmentation",
            "Inference",
            "Segmentation inference TensorFlow",
            "Alex Rozgo <alex.rozgo@gmail.com>",
        );

        // Create and add pad templates for our sink and source pad. These
        // are later used for actually creating the pads and beforehand
        // already provide information to GStreamer about all possible
        // pads that could exist for this type.

        // On the src pad, we can produce Rgb and Gray8 of any
        // width/height and with any framerate
        let caps = gst::Caps::new_simple(
            "video/x-raw",
            &[
                (
                    "format",
                    &gst::List::new(&[
                        &gst_video::VideoFormat::Rgb.to_string(),
                        &gst_video::VideoFormat::Gray8.to_string(),
                    ]),
                ),
                ("width", &gst::IntRange::<i32>::new(0, i32::MAX)),
                ("height", &gst::IntRange::<i32>::new(0, i32::MAX)),
                (
                    "framerate",
                    &gst::FractionRange::new(
                        gst::Fraction::new(0, 1),
                        gst::Fraction::new(i32::MAX, 1),
                    ),
                ),
            ],
        );
        // The src pad template must be named "src" for basetransform
        // and specific a pad that is always there
        let src_pad_template = gst::PadTemplate::new(
            "src",
            gst::PadDirection::Src,
            gst::PadPresence::Always,
            &caps,
        )
        .unwrap();
        klass.add_pad_template(src_pad_template);

        // On the sink pad, we can accept Rgb of any
        // width/height and with any framerate
        let caps = gst::Caps::new_simple(
            "video/x-raw",
            &[
                ("format", &gst_video::VideoFormat::Rgb.to_string()),
                ("width", &gst::IntRange::<i32>::new(0, i32::MAX)),
                ("height", &gst::IntRange::<i32>::new(0, i32::MAX)),
                (
                    "framerate",
                    &gst::FractionRange::new(
                        gst::Fraction::new(0, 1),
                        gst::Fraction::new(i32::MAX, 1),
                    ),
                ),
            ],
        );
        // The sink pad template must be named "sink" for basetransform
        // and specific a pad that is always there
        let sink_pad_template = gst::PadTemplate::new(
            "sink",
            gst::PadDirection::Sink,
            gst::PadPresence::Always,
            &caps,
        )
        .unwrap();
        klass.add_pad_template(sink_pad_template);

        // Install all our properties
        klass.install_properties(&PROPERTIES);

        // Configure basetransform so that we are never running in-place,
        // don't passthrough on same caps and also never call transform_ip
        // in passthrough mode (which does not matter for us here).
        //
        // We could work in-place for Rgb->Rgb but don't do here for simplicity
        // for now.
        klass.configure(
            gst_base::subclass::BaseTransformMode::NeverInPlace,
            false,
            false,
        );
    }
}

// Implementation of glib::Object virtual methods
impl ObjectImpl for Segmentation {
    // This macro provides some boilerplate.
    glib_object_impl!();

    // Called whenever a value of a property is changed. It can be called
    // at any time from any thread.
    fn set_property(&self, obj: &glib::Object, id: usize, value: &glib::Value) {
        let prop = &PROPERTIES[id];
        let element = obj.downcast_ref::<gst_base::BaseTransform>().unwrap();

        match *prop {
            subclass::Property("invert", ..) => {
                let mut settings = self.settings.lock().unwrap();
                let invert = value.get_some().unwrap();
                gst_info!(
                    self.cat,
                    obj: element,
                    "Changing invert from {} to {}",
                    settings.invert,
                    invert
                );
                settings.invert = invert;
            }
            subclass::Property("shift", ..) => {
                let mut settings = self.settings.lock().unwrap();
                let shift = value.get_some().unwrap();
                gst_info!(
                    self.cat,
                    obj: element,
                    "Changing shift from {} to {}",
                    settings.shift,
                    shift
                );
                settings.shift = shift;
            }
            _ => unimplemented!(),
        }
    }

    // Called whenever a value of a property is read. It can be called
    // at any time from any thread.
    fn get_property(&self, _obj: &glib::Object, id: usize) -> Result<glib::Value, ()> {
        let prop = &PROPERTIES[id];

        match *prop {
            subclass::Property("invert", ..) => {
                let settings = self.settings.lock().unwrap();
                Ok(settings.invert.to_value())
            }
            subclass::Property("shift", ..) => {
                let settings = self.settings.lock().unwrap();
                Ok(settings.shift.to_value())
            }
            _ => unimplemented!(),
        }
    }

    fn constructed(&self, obj: &glib::Object) {
        self.parent_constructed(obj);
    }
    
}

// Implementation of gst::Element virtual methods
impl ElementImpl for Segmentation {}

// Implementation of gst_base::BaseTransform virtual methods
impl BaseTransformImpl for Segmentation {
    // Called for converting caps from one pad to another to account for any
    // changes in the media format this element is performing.
    //
    // In our case that means that:
    fn transform_caps(
        &self,
        element: &gst_base::BaseTransform,
        direction: gst::PadDirection,
        caps: &gst::Caps,
        filter: Option<&gst::Caps>,
    ) -> Option<gst::Caps> {
        let other_caps = if direction == gst::PadDirection::Src {
            // For src to sink, no matter if we get asked for Rgb or Gray8 caps, we can only
            // accept corresponding Rgb caps on the sinkpad. We will only ever get Rgb and Gray8
            // caps here as input.
            let mut caps = caps.clone();

            for s in caps.make_mut().iter_mut() {
                s.set("format", &gst_video::VideoFormat::Rgb.to_string());
            }

            caps
        } else {
            // For the sink to src case, we will only get Rgb caps and for each of them we could
            // output the same caps or the same caps as Gray8. We prefer Gray8 (put it first), and
            // at a later point the caps negotiation mechanism of GStreamer will decide on which
            // one to actually produce.
            let mut gray_caps = gst::Caps::new_empty();

            {
                let gray_caps = gray_caps.get_mut().unwrap();

                for s in caps.iter() {
                    let mut s_gray = s.to_owned();
                    s_gray.set("format", &gst_video::VideoFormat::Gray8.to_string());
                    gray_caps.append_structure(s_gray);
                }
                gray_caps.append(caps.clone());
            }

            gray_caps
        };

        gst_debug!(
            self.cat,
            obj: element,
            "Transformed caps from {} to {} in direction {:?}",
            caps,
            other_caps,
            direction
        );

        // In the end we need to filter the caps through an optional filter caps to get rid of any
        // unwanted caps.
        if let Some(filter) = filter {
            Some(filter.intersect_with_mode(&other_caps, gst::CapsIntersectMode::First))
        } else {
            Some(other_caps)
        }
    }

    // Returns the size of one processing unit (i.e. a frame in our case) corresponding
    // to the given caps. This is used for allocating a big enough output buffer and
    // sanity checking the input buffer size, among other things.
    fn get_unit_size(&self, _element: &gst_base::BaseTransform, caps: &gst::Caps) -> Option<usize> {
        gst_video::VideoInfo::from_caps(caps).map(|info| info.size())
    }

    // Called whenever the input/output caps are changing, i.e. in the very beginning before data
    // flow happens and whenever the situation in the pipeline is changing. All buffers after this
    // call have the caps given here.
    //
    // We simply remember the resulting VideoInfo from the caps to be able to use this for knowing
    // the width, stride, etc when transforming buffers
    fn set_caps(
        &self,
        element: &gst_base::BaseTransform,
        incaps: &gst::Caps,
        outcaps: &gst::Caps,
    ) -> bool {
        let in_info = match gst_video::VideoInfo::from_caps(incaps) {
            None => return false,
            Some(info) => info,
        };
        let out_info = match gst_video::VideoInfo::from_caps(outcaps) {
            None => return false,
            Some(info) => info,
        };

        gst_debug!(
            self.cat,
            obj: element,
            "Configured for caps {} to {}",
            incaps,
            outcaps
        );

        *self.state.lock().unwrap() = Some(State { in_info, out_info });

        true
    }

    // Called when shutting down the element so we can release all stream-related state
    // There's also start(), which is called whenever starting the element again
    fn stop(&self, element: &gst_base::BaseTransform) -> Result<(), gst::ErrorMessage> {
        // Drop state
        let _ = self.state.lock().unwrap().take();

        gst_info!(self.cat, obj: element, "Stopped");

        Ok(())
    }

    // Does the actual transformation of the input buffer to the output buffer
    fn transform(
        &self,
        element: &gst_base::BaseTransform,
        inbuf: &gst::Buffer,
        outbuf: &mut gst::BufferRef,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        // Keep a local copy of the values of all our properties at this very moment. This
        // ensures that the mutex is never locked for long and the application wouldn't
        // have to block until this function returns when getting/setting property values
        // let settings = *self.settings.lock().unwrap();

        // Get a locked reference to our state, i.e. the input and output VideoInfo
        let mut state_guard = self.state.lock().unwrap();
        let state = state_guard.as_mut().ok_or_else(|| {
            gst_element_error!(element, gst::CoreError::Negotiation, ["Have no state yet"]);
            gst::FlowError::NotNegotiated
        })?;

        // Map the input buffer as a VideoFrameRef. This is similar to directly mapping
        // the buffer with inbuf.map_readable() but in addition extracts various video
        // specific metadata and sets up a convenient data structure that directly gives
        // pointers to the different planes and has all the information about the raw
        // video frame, like width, height, stride, video format, etc.
        //
        // This fails if the buffer can't be read or is invalid in relation to the video
        // info that is passed here
        let in_frame =
            gst_video::VideoFrameRef::from_buffer_ref_readable(inbuf.as_ref(), &state.in_info)
                .ok_or_else(|| {
                    gst_element_error!(
                        element,
                        gst::CoreError::Failed,
                        ["Failed to map input buffer readable"]
                    );
                    gst::FlowError::Error
                })?;

        // And now map the output buffer writable, so we can fill it.
        let mut out_frame =
            gst_video::VideoFrameRef::from_buffer_ref_writable(outbuf, &state.out_info)
                .ok_or_else(|| {
                    gst_element_error!(
                        element,
                        gst::CoreError::Failed,
                        ["Failed to map output buffer writable"]
                    );
                    gst::FlowError::Error
                })?;

        // Keep the various metadata we need for working with the video frames in
        // local variables. This saves some typing below.
        let width = in_frame.width() as usize;
        let in_stride = in_frame.plane_stride()[0] as usize;
        let in_data = in_frame.plane_data(0).unwrap();
        let out_stride = out_frame.plane_stride()[0] as usize;
        let out_format = out_frame.format();
        let out_width = out_frame.width() as i32;
        let out_height = out_frame.height() as i32;
        let out_data = out_frame.plane_data_mut(0).unwrap();

        if out_format == gst_video::VideoFormat::Rgb {
            assert_eq!(in_data.len() % 3, 0);
            assert_eq!(out_data.len() % 3, 0);
            assert_eq!(out_data.len() / out_stride, in_data.len() / in_stride);

            let in_line_bytes = width * 3;
            let out_line_bytes = width * 3;

            assert!(in_line_bytes <= in_stride);
            assert!(out_line_bytes <= out_stride);

            let mut mut_in_data = in_data.to_vec();

            let mut src = Mat::new_rows_cols_with_data(
                    out_height,
                    out_width,
                    opencv::core::CV_8UC3,
                    unsafe { transmute(mut_in_data.as_mut_ptr()) },
                    in_stride
                )
                .unwrap();


            let mut detector = objdetect::QRCodeDetector::default().unwrap();
            let mut pts = VectorOfPoint::new();
            let mut straight = Mat::default().unwrap();
            let decoded_data = detector.detect_and_decode(&src, &mut pts, &mut straight).unwrap();

            if !straight.empty().unwrap() {
                let p_to_slice = pts.to_slice();
                let qr_rect = Rect::from_points(p_to_slice[0], p_to_slice[2]);
                let rect_color = Scalar::new(255., 0., 0., 1.);

                imgproc::rectangle(&mut src, qr_rect, rect_color, 2, 8, 0).unwrap();

                // println!("{:?} - Decoded Data: {}", p_to_slice, decoded_data);
            }
            
            let src = src.reshape(1, 1).unwrap();

            for i in 0 .. src.size().unwrap().width {
                out_data[i as usize] = *src.at::<u8>(i).unwrap();
            }

        } else if out_format == gst_video::VideoFormat::Gray8 {
            unimplemented!();
        } else {
            unimplemented!();
        }

        Ok(gst::FlowSuccess::Ok)
    }
}

// Registers the type for our element, and then registers in GStreamer under
// the name "segmentation" for being able to instantiate it via e.g.
// gst::ElementFactory::make().
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "tf_segmentation",
        gst::Rank::None,
        Segmentation::get_type(),
    )
}
