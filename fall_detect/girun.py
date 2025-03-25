import gi
import os
from gi.repository import Gst, GstApp, GLib


class grn:
    def __init__(self, using_pipeline = "testsink"): # GST init
        gi.require_version('Gst','1.0')
        
        Gst.init(None)

        gst_is = f"appsrc name=source is-live=true format=time ! videoconvert ! video/x-raw, format=I420 , width=640, height=640 !"
        f" x264enc byte-stream=true tune=zerolatency bitrate=500 speed-preset=ultrafast ! video/x-h264, level=4 ! h264parse ! video/x-h264, stream-format=avc, alignment=au !"
        f" kvssink stream-name=STREAM access-key={os.environ['AWS_ACCESS_KEY_ID']} secret-key={os.environ['AWS_SECRET_ACCESS_KEY']} aws-region={os.environ['AWS_DEFAULT_REGION']}"

        gst_is2=f"appsrc name=source is-live=true format=time ! videoconvert ! video/x-raw, format=I420 , width=640, height=640 !"
        f" x264enc byte-stream=true tune=zerolatency bitrate=500 speed-preset=ultrafast ! video/x-h264, level=4 ! h264parse ! video/x-h264, stream-format=avc, alignment=au ! filesink location=test.mkv"

        gst_is3 = f"appsrc name=source is-live=true format=time ! videoconvert ! video/x-raw, format=I420 , width=640, height=640 ! filesink location=test2.yuv"

        gst_is4 = f"appsrc name=source is-live=true format=time ! videoconvert ! video/x-raw, format=I420 , width=640, height=640 !"
        f" x264enc byte-stream=true tune=zerolatency bitrate=500 ! matroskamux ! filesink location=test3.mkv"

        gst_is5 = f"appsrc name=source is-live=true format=time do-timestamp=true ! videoconvert !"
        f" video/x-raw, format=I420 , width=640, height=640 ! x264enc byte-stream=true tune=zerolatency bitrate=500 key-int-max=7 !"
        f" kvssink stream-name=STREAM access-key={os.environ['AWS_ACCESS_KEY_ID']} secret-key={os.environ['AWS_SECRET_ACCESS_KEY']} aws-region={os.environ['AWS_DEFAULT_REGION']}" 

        testsink = f"videotestsrc is-live=true ! videoconvert ! video/x-raw, format=RGB, width=640, height=640 ! x264enc tune=zerolatency bitrate=500 ! video/x-h264, stream-format=avc, alignment=au !"
        f" kvssink stream-name=STREAM access-key={os.environ['AWS_ACCESS_KEY_ID']} secret-key={os.environ['AWS_SECRET_ACCESS_KEY']} aws-region={os.environ['AWS_DEFAULT_REGION']}"

        self.pipeline = None
        if(using_pipeline == "gst_is"):
            self.pipeline= Gst.parse_launch(gst_is)
        if(using_pipeline == "gst_is2"):
            self.pipeline= Gst.parse_launch(gst_is2)
        if(using_pipeline == "gst_is3"):
            self.pipeline= Gst.parse_launch(gst_is3)
        if(using_pipeline == "gst_is4"):
            self.pipeline= Gst.parse_launch(gst_is4)
        if(using_pipeline == "gst_is5"):
            self.pipeline= Gst.parse_launch(gst_is5)
        if(using_pipeline == "testsink"):
            self.pipeline= Gst.parse_launch(testsink)
        
        self.appsrc = None
        caps = Gst.Caps.from_string("video/x-raw, format=BGR, width=640, height=640")

        if(using_pipeline != "testsink"):
            self.appsrc=self.pipeline.get_by_name("source")
            self.appsrc.set_property("format", Gst.Format.TIME)
            self.appsrc.set_property("is-live",True)
            self.appsrc.set_property("block",True) # 버퍼가 찬 경우 블록
            self.appsrc.set_property("max-bytes",2*1024*1024)
            self.appsrc.set_property("do-timestamp",True)

            self.appsrc.set_caps(caps)

        self.pipeline.set_state(Gst.State.PLAYING)
    
    def push_data(self, input):
        inpd= input.tobytes()
        Buffer = Gst.Buffer.new_wrapped(inpd)
        if self.appsrc is not None:
            ret = self.appsrc.emit("push-buffer", Buffer)
            if ret != Gst.FlowReturn.OK:
                print(f"Error Pushing Buffer: {ret}")
                return False
        return True
    
    def end_stream(self):
        self.appsrc.emit('end-of-stream')
        self.pipeline.set_state(Gst.State.NULL)
