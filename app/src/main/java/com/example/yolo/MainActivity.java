package com.example.yolo;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Environment;
import android.view.SurfaceView;
import android.view.View;
import android.view.animation.AlphaAnimation;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.NumberPicker;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import com.physicaloid.lib.Physicaloid;
import com.physicaloid.lib.usb.driver.uart.ReadLisener;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;
    Net tinyYolo;
    TextView confidenceView;

    Button startBut;
    boolean startYolo = false, firstTimeYolo = false;

    private AlphaAnimation buttonClick = new AlphaAnimation(2F, 0.2F);


    public void YOLO(View Button){
        Button.startAnimation(buttonClick);
        if (startYolo == false){
            startYolo = true;
            startBut.setText("STOP");
            if (firstTimeYolo == false)
            {
                firstTimeYolo = true;
                String tinyYoloCfg = Environment.getExternalStorageDirectory() + "/dnn/yolov3-tiny.cfg" ;
                String tinyYoloWeights = Environment.getExternalStorageDirectory() + "/dnn/yolov3-tiny.weights";
                System.out.println(tinyYoloCfg);
                System.out.println(tinyYoloWeights);
                tinyYolo = Dnn.readNetFromDarknet(tinyYoloCfg, tinyYoloWeights);
            }
        }
        else
            {
            startBut.setText("START");
            startYolo = false;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.CameraView);
        cameraBridgeViewBase.setMaxFrameSize(1280,960);//500-500
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);

        startBut = findViewById(R.id.button3);
        confidenceView =  findViewById(R.id.confidenceView);



        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);
                switch(status){
                    case BaseLoaderCallback.SUCCESS:
                        cameraBridgeViewBase.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }
            }
        };
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat frame = inputFrame.rgba();
        if (startYolo) {
            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
            Mat imageBlob = Dnn.blobFromImage(frame, 0.00392, new Size(416,416),new Scalar(0, 0, 0),false, false);
            tinyYolo.setInput(imageBlob);
            List<Mat> result = new ArrayList<Mat>(2);

            List<String> outBlobNames = new ArrayList<>();
            outBlobNames.add(0, "yolo_16");
            outBlobNames.add(1, "yolo_23");

            tinyYolo.forward(result,outBlobNames);

            float confThreshold = 0.5f;

            List<Integer> clsIds = new ArrayList<>();
            List<Float> confs = new ArrayList<>();
            List<Rect> rects = new ArrayList<>();

            int centerX =0;
            int centerY=0;

            for (int i = 0; i < result.size(); ++i)
            {
                Mat level = result.get(i);
                for (int j = 0; j < level.rows(); ++j)
                {
                    Mat row = level.row(j);
                    Mat scores = row.colRange(5, level.cols());
                    Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                    float confidence = (float)mm.maxVal;

                    Point classIdPoint = mm.maxLoc;
                    if (confidence > confThreshold)
                    {
                        centerX = (int)(row.get(0,0)[0] * frame.cols());
                        centerY = (int)(row.get(0,1)[0] * frame.rows());
                        int width   = (int)(row.get(0,2)[0] * frame.cols());
                        int height  = (int)(row.get(0,3)[0] * frame.rows());
                        int left    = centerX - width  / 2;
                        int top     = centerY - height / 2;
                        clsIds.add((int)classIdPoint.x);
                        confs.add((float)confidence);
                        rects.add(new Rect(left, top, width, height));

                    }
                }
            }
            int ArrayLength = confs.size();

            if (ArrayLength>=1) {
                float nmsThresh = 0.2f;
                MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
                Rect[] boxesArray = rects.toArray(new Rect[0]);
                MatOfRect boxes = new MatOfRect(boxesArray);
                MatOfInt indices = new MatOfInt();
                Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);

                int[] ind = indices.toArray();
                int intConf;

               for (int i = 0; i < ind.length; ++i)
                {
                    int idx = ind[i];
                    Rect box = boxesArray[idx];
//                    int idGuy = clsIds.get(idx);
                    float conf = confs.get(idx);
//                    List<String> cocoNames = Arrays.asList("Rugby_ball");
                    intConf = (int) (conf * 100);

                    final int finalIntConf = intConf;
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            confidenceView.setText(Integer.toString(finalIntConf)+"%");
                        }
                    });

                    Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(255, 0, 0), 3);
                }

            }
        }

        return frame;
    }


    @Override
    public void onCameraViewStarted(int width, int height) {
        if (startYolo == true)
        {
            String tinyYoloCfg = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny.cfg" ;
            String tinyYoloWeights = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny.weights";
            tinyYolo = Dnn.readNetFromDarknet(tinyYoloCfg, tinyYoloWeights);
        }
        }

        @Override
        public void onCameraViewStopped() {
        }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(),"There's a problem, yo!", Toast.LENGTH_SHORT).show();
        }
        else
        {
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }
}