//
//  video_capture.cpp
//  videoPro
//
//  Created by lingLong on 16/4/22.
//  Copyright © 2016年 ling. All rights reserved.
//

#include "video_capture.hpp"




void canny(cv::Mat &img,cv::Mat &out)
{
    cv::Mat tmp(img.size(),img.type());
    if(3 == img.channels())
        cvtColor(img, tmp, CV_BGR2GRAY);
    Canny(tmp, out, 100, 200);
    threshold(out, out, 128, 255, THRESH_BINARY_INV);
}



//int camTrack::cam_tracker(string filename)
//{
//    VideoCapture cap(filename);
//    Rect trackWindow;
//    int hsize = 16;
//    float hranges[] = {0,180};
//    const float* phranges = hranges;
//    //    CommandLineParser parser(argc, argv, keys);
//    //    if (parser.has("help"))
//    //    {
//    //        help();
//    //        return 0;
//    //    }
//    //    int camNum = parser.get<int>(0);
//    //    cap.open(camNum);
//    
//    if( !cap.isOpened() )
//    {
//        help();
//        cout << "***Could not initialize capturing...***\n";
//        cout << "Current parameter's value: \n";
//        return -1;
//    }
//    cout << hot_keys;
//    namedWindow( "Histogram", 0 );
//    namedWindow( "CamShift Demo", 0 );
//    setMouseCallback( "CamShift Demo", onMouse, 0 );
//    createTrackbar( "Vmin", "CamShift Demo", &vmin, 256, 0 );//最后一个参数为0代表没有调用滑动拖动的响应函数
//    createTrackbar( "Vmax", "CamShift Demo", &vmax, 256, 0 );
//    createTrackbar( "Smin", "CamShift Demo", &smin, 256, 0 );
//    
//    Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
//    bool paused = false;
//    
//    for(;;)
//    {
//        if( !paused )
//        {
//            cap >> frame;
//            if( frame.empty() )
//                break;
//        }
//        
//        frame.copyTo(image);
//        
//        if( !paused )
//        {
//            cvtColor(image, hsv, COLOR_BGR2HSV);
//            
//            if( trackObject )//trackObject初始化为0,或者按完键盘的'c'键后也为0，当鼠标单击松开后为-1
//            {
//                int _vmin = vmin, _vmax = vmax;
//                //inRange函数的功能是检查输入数组每个元素大小是否在2个给定数值之间，可以有多通道,mask保存0通道的最小值，也就是h分量
//                //这里利用了hsv的3个通道，比较h,0~180,s,smin~256,v,min(vmin,vmax),max(vmin,vmax)。如果3个通道都在对应的范围内，则
//                //mask对应的那个点的值全为1(0xff)，否则为0(0x00).
//                inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
//                        Scalar(180, 256, MAX(_vmin, _vmax)), mask);
//                int ch[] = {0, 0};
//                hue.create(hsv.size(), hsv.depth());//hue初始化为与hsv大小深度一样的矩阵，色调的度量是用角度表示的，红绿蓝之间相差120度，反色相差180度
//                mixChannels(&hsv, 1, &hue, 1, ch, 1);//将hsv第一个通道(也就是色调)的数复制到hue中，0索引数组
//                if( trackObject < 0 )//鼠标选择区域松开后，该函数内部又将其赋值-1
//                {
//                    Mat roi(hue, selection), maskroi(mask, selection);//此处的构造函数roi用的是Mat hue的矩阵头，且roi的数据指针指向hue，即共用相同的数据，select为其感兴趣的区域
//                    calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
//                    normalize(hist, hist, 0, 255, NORM_MINMAX);
//                    
//                    trackWindow = selection;
//                    trackObject = 1;//只要鼠标选完区域松开后，且没有按键盘清0键'c'，则trackObject一直保持为1，因此该if函数只能执行一次，除非重新选择跟踪区域
//                    histimg = Scalar::all(0);
//                    int binW = histimg.cols / hsize;
//                    Mat buf(1, hsize, CV_8UC3);//缓冲用单bin
//                    for( int i = 0; i < hsize; i++ )
//                        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
//                    cvtColor(buf, buf, COLOR_HSV2BGR);
//                    
//                    for( int i = 0; i < hsize; i++ )
//                    {
//                        int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
//                        rectangle( histimg, Point(i*binW,histimg.rows),
//                                  Point((i+1)*binW,histimg.rows - val),
//                                  Scalar(buf.at<Vec3b>(i)), -1, 8 );
//                    }
//                }
//                
//                calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
//                backproj &= mask;
//                RotatedRect trackBox = CamShift(backproj, trackWindow,TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
//                if( trackWindow.area() <= 1 )
//                {
//                    int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
//                    trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
//                                       trackWindow.x + r, trackWindow.y + r) &
//                    Rect(0, 0, cols, rows);
//                }
//                
//                if( backprojMode )
//                    cvtColor( backproj, image, COLOR_GRAY2BGR );
//                ellipse( image, trackBox, Scalar(0,255,255), 3, LINE_AA );
//                //                Rect rect = trackBox.boundingRect();
//                //                rectangle(image, rect, Scalar(0,255,0),5,LINE_AA);
//            }
//        }
//        else if( trackObject < 0 )
//            paused = false;
//        
//        if( selectObject && selection.width > 0 && selection.height > 0 )
//        {
//            Mat roi(image, selection);
//            bitwise_not(roi, roi);
//        }
//        
//        imshow( "CamShift Demo", image );
//        imshow( "Histogram", histimg );
//        
//        char c = (char)waitKey(10);
//        if( c == 27 )
//            break;
//        switch(c)
//        {
//            case 'b':
//                backprojMode = !backprojMode;
//                break;
//            case 'c':
//                trackObject = 0;
//                histimg = Scalar::all(0);
//                break;
//            case 'h':
//                showHist = !showHist;
//                if( !showHist )
//                    destroyWindow( "Histogram" );
//                else
//                    namedWindow( "Histogram", 1 );
//                break;
//            case 'p':
//                paused = !paused;
//                break;
//            default:
//                ;
//        }
//    }
//    return 0;
//    
//}


