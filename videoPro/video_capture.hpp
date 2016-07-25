//
//  video_capture.hpp
//  videoPro
//
//  Created by lingLong on 16/4/22.
//  Copyright © 2016年 ling. All rights reserved.
//

#ifndef video_capture_hpp
#define video_capture_hpp

#include <stdio.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ctype.h>

using namespace std;
using namespace cv;




//帧处理类
class frameProcessor{
public:
    virtual void process(cv::Mat &input,cv::Mat &output) = 0;
    
    
};



class VideoProcessor
{
private:
    cv::VideoCapture capture;//video cpature object
    void(*process)(cv::Mat &,cv::Mat &);//callback function of every frame calls
    bool call_it;//确定是否调用回调函数的bool变量
    string windowNameInput;
    string windowNameOutput;
    int delay;
    long fnumber;//已处理的帧数
    long frame_to_stop;//在该帧数停止
    bool stop;//是否停止处理
    vector<string> images;
    vector<string>::const_iterator imgIt;
    //opencv的写视频对象
    VideoWriter writer;
    //输出文件名称
    string outputfile;
    //输出文件当前索引
    int current_index;
    //输出图像名称的位数
    int digits;
    //输出图像的扩展名
    string extension;
    //制定跟踪的书脊矩形框
    Rect pr;
public:
    //initialize
    VideoProcessor():call_it(true),delay(0),fnumber(0),frame_to_stop(-1),stop(false) {}
//    void canny(cv::Mat &img,cv::Mat &out);
//    设置frameProcessor实例
    frameProcessor *frameprocess;
    void set_frame_processor(frameProcessor *fptr)
    {
        //使回调函数无效
        process = 0;
        //重新设置实例
        
        frameprocess = fptr;
        callProcess();
        
        
    }
    
    //设置回调函数
    void set_frame_processor(void(*frame_processing_callback) (cv::Mat &,cv::Mat &))
    {
        frameprocess = 0;
        process = frame_processing_callback;
        callProcess();
    }
    //设置视频文件的名称
    bool set_input(string filename)
    {
        fnumber = 0;
        capture.release();
        images.clear();
        return capture.open(filename);
    }
    //设置输入的图像向量
    bool set_input(vector<string> &imgs)
    {
        fnumber = 0;
        //释放之前打开过的资源
        capture.release();
        //输入将是该图像的向量
        images = imgs;
        imgIt = images.begin();
        return  true;
    }
    
    //设置输出为视频文件，默认使用与输入视频相同的参数
    bool set_output(const string &filename,int codec = 0,double fps = 0.0,bool is_color = true)
    {
        outputfile = filename;
        extension.clear();
        if(fps == 0.0)fps = get_frame_rate();
        char c[4];
        if(0 == codec )codec = get_codec(c);
//        Size size = capture.get(cv_cap_pro_fra)
        return writer.open(outputfile, codec, fps, get_frame_size(),is_color);
    }
    Size get_frame_size()
    {
        int h = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
        int w = capture.get(CV_CAP_PROP_FRAME_WIDTH);
        return Size(w,h);
        
    }
    long get_frame_rate()
    {
        return capture.get(CV_CAP_PROP_FPS);
    }
    
    //获取输入视频的编码方式
    int get_codec(char codec[4])
    {
        if(images.size() != 0)return -1;
        union{
            //4-char的数据编码结果
            int value;
            char code[4];
        }returned;
        returned.value = (int)capture.get(CV_CAP_PROP_FOURCC);//得到编码
        codec[0] = returned.code[0];
        codec[1] = returned.code[1];
        codec[2] = returned.code[2];
        codec[3] = returned.code[3];
        return returned.value;
    }
    //保存输出帧，可能是：视频或图像
    void write_next_frame(cv::Mat &frame)
    {
        if(extension.length())//输出到图像文件
        {
            stringstream ss;
            ss<<outputfile<<std::setfill('0')
              <<std::setw(digits)
            <<current_index++<<extension;
            imwrite(ss.str(), frame);
        }
        else//输出到视频文件
        {
//            cv::Mat tmp();
            if(!frame.empty())
            {
                if(frame.channels() != 3 || frame.channels() != 4)
                    cvtColor(frame, frame, CV_GRAY2BGR);
                writer.write(frame);
            }
        }
    }
    
    //设置输出为独立的图像文件，扩展名必须是jpg，bmp
    bool set_output(const string &filename,const string &ext,int number_of_digits = 3,int start_index = 0)
    {
        if(number_of_digits < 0)
            return false;
        outputfile = filename;
        extension = ext;
        digits = number_of_digits;
        current_index = start_index;
        return true;
    }
   //捕捉视频设备是否已经打开
    bool is_opened()
    {
        return capture.isOpened() || !images.empty();
    }
    //create input window
    void display_input(string window)
    {
        windowNameInput = window;
        namedWindow(windowNameInput);
    }
    //create input window
    void display_output(string window)
    {
        windowNameOutput = window;
        namedWindow(windowNameOutput);
    }
    //不在显示处理后的帧
    void dont_display()
    {
        destroyWindow(windowNameInput);
        destroyWindow(windowNameOutput);
        windowNameInput.clear();
        windowNameOutput.clear();
    }
    //得到下一帧，可以是：视频文件，摄像头，图像数组
    bool read_next_frame(cv::Mat &frame)
    {
        if(!images.size())
            return capture.read(frame);
        else
        {
            if(imgIt != images.end())
            {
                frame = imread(*imgIt);
//                cout<<*imgIt<<endl;
//                display_output("first frame");
//                imshow(windowNameOutput, frame);
                imgIt ++;
                return frame.data != 0;
            }
            else
                return false;
        }
    }
    
    
    //获取并处理序列帧
    void run()
    {
        //current frame
        cv::Mat frame;
        //ouput frame
        cv::Mat output;
        if (! is_opened())
        {
            return;
        }
        stop = false;
        while (! is_stopped())
        {
            //读取下一帧
            if(! read_next_frame(frame))
                break;
//            display_input("display a frame");
            if(windowNameInput.length() != 0)
                imshow(windowNameInput, frame);
            //调用处理函数
            if(call_it)
            {
                if(process)
                {
                    //处理当前帧
                    process(frame,output);
                }
                else if (frameprocess)
                    frameprocess->process(frame, output);
                fnumber ++;
                
            }
            else
            {
                output = frame;
            }
            //输出图像序列
            if(outputfile.length())
                write_next_frame(output);
            //显示输出帧
//            display_output("display out frame");
            if( windowNameOutput.length() != 0)
                imshow(windowNameOutput, output);
//            imshow("display out frame", output);
            //引入延迟
            if(delay >= 0 && waitKey(delay) >= 0 )
                stopIt();//停止运行
            if(frame_to_stop >= 0 && get_frame_number() == frame_to_stop)
                stopIt();
        }
    }
        //停止运行
    void stopIt()
    {
        stop = true;
    }
    //是否已经停止
    bool is_stopped()
    {
        return stop;
    }
    
    //set the delay,0 means waiting for the user to press key,
    void set_delay(int d)
    {
        delay = d;
    }
    //需要调用回调函数
    void callProcess()
    {
        call_it = true;
    }
    //不需要调用回调函数
    void not_callProcess()
    {
        call_it = false;
    }
    //在指定帧停止
    void stop_frame_num(long frame)
    {
        frame_to_stop = frame;
    }
    //返回一帧的帧数
    long get_frame_number()
    {
        return capture.get(CV_CAP_PROP_POS_FRAMES);
    }
    
    
    
};


class FeatureTracker:public frameProcessor{
    cv::Mat gray_curr;
    cv::Mat gray_prev;//之前的灰度图像
    //两幅图像间跟踪的特征点 0->1
    vector<cv::Point2f> points[2];
    //跟踪的初始点位置
    vector<cv::Point2f> initial;
    //特征点
    vector<cv::Point2f> features;
    //需要跟踪的最大特征数
    int max_count;
    //评估跟踪的质量
    double q_lel;
    //两点之间的最小距离
    double min_dist;
    //跟踪状态
    vector<uchar> status;
    //跟踪过程中的error
    vector<float> err;
public:
    FeatureTracker():max_count(500),q_lel(0.01),min_dist(10.) {}
    //该方法将在每一帧被调用，首先，在需要时检测特征点，接着，跟踪这些点，将无法跟踪或是不再希望跟踪的点剔除掉，最后当前帧以及它的点在下一次迭代中成为之前的帧以及之前的点
    void process(cv::Mat &frame,cv::Mat & output)
    {
        cvtColor(frame, gray_curr, CV_BGR2GRAY);
        frame.copyTo(output);
        //1.如果需要添加新的特征点
        if(add_new_points())
        {
            //检测新的特征点
            detect_features();
            //添加到当前跟踪的特征中
            points[0].insert(points[0].end(), features.begin(), features.end());
            initial.insert(initial.end(), features.begin(), features.end());
            
        }
        //对于序列中的第一幅图像
        if(gray_prev.empty())
            gray_curr.copyTo(gray_prev);
        //2.跟踪特征点
        calcOpticalFlowPyrLK(gray_prev, gray_curr, points[0], points[1], status, err);
        //2.遍历所有跟踪的点进行筛选
        int k = 0;
        for (int i = 0; i < points[1].size(); i ++)
        {
            //是否保留
            if(accept_tracked_ponints(i))
            {
                initial[k] = initial[i];
                points[1][k++] = points[1][i];
            }
        }
        //剔除跟踪不成功的点
        points[1].resize(k);
        initial.resize(k);
        //3.处理接受的跟踪点
        handle_tracked_points(frame, output);
        //4.当前帧的点和图像变为前一帧的点和图像
        swap(points[1], points[0]);
        swap(gray_prev,gray_curr);
        
    }
    
    
    //是否添加新的特征点
    bool add_new_points()
    {
        //点的数量太少
        return points[0].size() <= 10;
    }
    //检测特征点
    void detect_features()
    {
        goodFeaturesToTrack(gray_curr, features, max_count, q_lel, min_dist);
    }
    //决定哪些点应该跟踪,剔除不在移动的点，以及calcOpticalFlowPyrLK无法跟踪的点
    bool accept_tracked_ponints(int i)
    {
        return status[i] &&
        //表示移动了
        ((abs(points[0][i].x - points[1][i].x) + abs(points[0][i].y - points[1][i].y)) > 2);
        
    }
    //处理当前跟踪的点
    void handle_tracked_points(cv::Mat &frame,cv::Mat &out)
    {
        //遍历所有跟踪点
        for (int i = 0; i < points[1].size(); i ++)
        {
            line(out, initial[i], points[1][i], Scalar(0,255,0));
            circle(out, points[1][i], 3, Scalar(255,255,255),-1);
        }
    }
    
    
};


class BGFGSegmentor:public frameProcessor{
    cv::Mat gray;//当前灰度图
    cv::Mat background;//累积的背景
    cv::Mat back_image;//背景图像
    cv:: Mat fore_image;//前景图像
    //背景累加中的学习率
    double lr;
    int threshold;//前景提取的阈值
    //制定跟踪的书脊矩形框
    Rect pr;
public:
    BGFGSegmentor():threshold(10),lr(0.01) {}
    //处理方法
    void process(cv::Mat &frame,cv::Mat &out)
    {
        cvtColor(frame, gray, CV_BGR2GRAY);
        //初始化背景为第一帧
        if(background.empty())
            gray.convertTo(background,CV_32F);
        background.convertTo(back_image, CV_8U);
        //计算差值
//        Ptr<BackgroundSubtractorMOG2> mog = createBackgroundSubtractorMOG2(10,10,true);
        absdiff(back_image, gray, fore_image);
//        mog->apply(frame,fore_image,0.01);
//        out = fore_image.clone();
        cv::threshold(fore_image, out, threshold, 255, CV_THRESH_BINARY_INV);
        //对背景累加
        accumulateWeighted(gray, background, lr,out);
        vector<vector<cv::Point>> contour;
        findContours(out, contour, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
        for (int i = 0; i < contour.size(); i ++)
        {
            Rect rect = boundingRect(contour[i]);
            double area = contourArea(contour[i]);
            if (area > 3000 && area < 7000) {
                rectangle(frame, rect, Scalar(0,255,0));
//                imshow("rect", frame);

            }

        }
//        rectangle(frame,pr,Scalar(0,255,0));
        out = frame.clone();
        
        
    }
    void set_thresh(int thresh)
    {
        threshold = thresh;
    }
    void  set_pr(vector<cv::Point> corner)
    {
        pr = boundingRect(corner);
    }
};

class camTrack{
private:
    //处理的当前帧图像
    cv::Mat image;
    //表示是否要进入反向投影模式
    bool backprojMode = false;
    //代表是否在选要跟踪的初始目标
    bool selectObject = false;
    //代表跟踪目标数目
    int trackObject = 0;
    //是否显示直方图
    bool showHist = true;
    //用于保存鼠标选择第一次单击时点的位置
    cv::Point origin;
    //用于保存鼠标选择的矩形框
    Rect selection;
    //视频窗口的滑动条的最小值和最大值，初始值
    int vmin = 10, vmax = 256, smin = 30;
    //热键对应操作
    
public:
    string hot_keys =
    "\n\nHot keys: \n"
    "\tESC - quit the program\n"
    "\tc - stop the tracking\n"
    "\tb - switch to/from backprojection view\n"
    "\th - show/hide object histogram\n"
    "\tp - pause video\n"
    "To initialize tracking, select the object with mouse\n";

    //鼠标点击事件
    static void onMouse( int event, int x, int y, int, void* )
    {
        camTrack ct;
        switch( event )
        {
            case EVENT_LBUTTONDOWN:
                ct.origin = cv::Point(x,y);
                ct.selection = Rect(x,y,0,0);
                ct.selectObject = true;
                break;
            case EVENT_LBUTTONUP:
                ct.selectObject = false;
                if( ct.selection.width > 0 && ct.selection.height > 0 )
                    ct.trackObject = -1;
                break;
        }
        
        if( ct.selectObject )
        {
            ct.selection.x = MIN(x, ct.origin.x);
            ct.selection.y = MIN(y, ct.origin.y);
            ct.selection.width = std::abs(x - ct.origin.x);
            ct.selection.height = std::abs(y - ct.origin.y);
            
            ct.selection &= Rect(0, 0, ct.image.cols, ct.image.rows);
        }
        
    }
    
    //help
    static void help()
    {
        camTrack ct;
        cout << "\nThis is a demo that shows mean-shift based tracking\n"
        "You select a color objects such as your face and it tracks it.\n"
        "This reads from video camera (0 by default, or the camera number the user enters\n"
        "Usage: \n"
        "   ./camshiftdemo [camera number]\n";
        cout << ct.hot_keys;
    }

    //利用camshift跟踪
    int cam_tracker(string filename)
    {
        VideoCapture cap(filename);
        Rect trackWindow;
        int hsize = 16;
        float hranges[] = {0,180};
        const float* phranges = hranges;
        //    CommandLineParser parser(argc, argv, keys);
        //    if (parser.has("help"))
        //    {
        //        help();
        //        return 0;
        //    }
        //    int camNum = parser.get<int>(0);
        //    cap.open(camNum);
        
        if( !cap.isOpened() )
        {
            help();
            cout << "***Could not initialize capturing...***\n";
            cout << "Current parameter's value: \n";
            //        parser.printMessage();
            return -1;
        }
        cout << hot_keys;
        namedWindow( "Histogram", 0 );
        namedWindow( "CamShift Demo", 0 );
        setMouseCallback( "CamShift Demo", onMouse, 0);
        createTrackbar( "Vmin", "CamShift Demo", &vmin, 256, 0 );
        createTrackbar( "Vmax", "CamShift Demo", &vmax, 256, 0 );
        createTrackbar( "Smin", "CamShift Demo", &smin, 256, 0 );
        
        Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
        bool paused = false;
        
        for(;;)
        {
            if( !paused )
            {
                cap >> frame;
                if( frame.empty() )
                    break;
            }
            
            frame.copyTo(image);
            
            if( !paused )
            {
                cvtColor(image, hsv, COLOR_BGR2HSV);
                
                if( trackObject )
                {
                    int _vmin = vmin, _vmax = vmax;
                    
                    inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
                            Scalar(180, 256, MAX(_vmin, _vmax)), mask);
                    int ch[] = {0, 0};
                    hue.create(hsv.size(), hsv.depth());
                    mixChannels(&hsv, 1, &hue, 1, ch, 1);
                    
                    if( trackObject < 0 )
                    {
                        Mat roi(hue, selection), maskroi(mask, selection);
                        calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                        normalize(hist, hist, 0, 255, NORM_MINMAX);
                        
                        trackWindow = selection;
                        trackObject = 1;
                        
                        histimg = Scalar::all(0);
                        int binW = histimg.cols / hsize;
                        Mat buf(1, hsize, CV_8UC3);
                        for( int i = 0; i < hsize; i++ )
                            buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
                        cvtColor(buf, buf, COLOR_HSV2BGR);
                        
                        for( int i = 0; i < hsize; i++ )
                        {
                            int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
                            rectangle( histimg, Point(i*binW,histimg.rows),
                                      Point((i+1)*binW,histimg.rows - val),
                                      Scalar(buf.at<Vec3b>(i)), -1, 8 );
                        }
                    }
                    
                    calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
                    backproj &= mask;
                    RotatedRect trackBox = CamShift(backproj, trackWindow,
                                                    TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
                    if( trackWindow.area() <= 1 )
                    {
                        int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
                        trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                           trackWindow.x + r, trackWindow.y + r) &
                        Rect(0, 0, cols, rows);
                    }
                    
                    if( backprojMode )
                        cvtColor( backproj, image, COLOR_GRAY2BGR );
                    ellipse( image, trackBox, Scalar(0,0,255), 3, LINE_AA );
                }
            }
            else if( trackObject < 0 )
                paused = false;
            
            if( selectObject && selection.width > 0 && selection.height > 0 )
            {
                Mat roi(image, selection);
                bitwise_not(roi, roi);
            }
            
            imshow( "CamShift Demo", image );
            imshow( "Histogram", histimg );
            
            char c = (char)waitKey(10);
            if( c == 27 )
                break;
            switch(c)
            {
                case 'b':
                    backprojMode = !backprojMode;
                    break;
                case 'c':
                    trackObject = 0;
                    histimg = Scalar::all(0);
                    break;
                case 'h':
                    showHist = !showHist;
                    if( !showHist )
                        destroyWindow( "Histogram" );
                    else
                        namedWindow( "Histogram", 1 );
                    break;
                case 'p':
                    paused = !paused;
                    break;
                default:
                    ;
            }
        }
        
        return 0;
    }

    

    
};


#endif /* video_capture_hpp */
