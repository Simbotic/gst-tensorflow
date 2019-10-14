#!/usr/bin/env bash

export GST_PLUGIN_PATH=$(pwd)/target/debug
# export GST_PLUGIN_PATH=`pwd`/target/release

cargo build && \
    gst-launch-1.0 filesrc location=assets/qr_obst.mp4 \
    ! decodebin \
    ! videoconvert \
    ! videoscale \
    ! qrcode_detector \
    ! video/x-raw,format=RGB,width=512,height=288 \
    ! videoconvert \
    ! ximagesink
# cargo build --release && gst-launch-1.0 filesrc location=assets/qr_obst.mp4 ! decodebin ! videoconvert ! videoscale ! tf_segmentation ! video/x-raw,format=RGB,width=512,height=288 ! videoconvert ! ximagesink
# cargo build && gst-launch-1.0 filesrc location=assets/sample.webm ! decodebin ! videoconvert ! videoscale ! rsqrcode ! videoconvert ! ximagesink

# cargo build && gst-launch-1.0 playbin uri=file://$(pwd)/assets/sample.webm video-filter=rsrgb2gray