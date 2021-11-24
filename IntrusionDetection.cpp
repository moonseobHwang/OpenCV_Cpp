/*
* Intrusion Detection Project

* 1.input image
* 입력영상이 저장영상,캠으로부터 받아오기

* 2.BG Subtraction
* 배경 제거

* 3.noise remove
* 잡음 제거
* Morphology 
* 1. Erosion 침식
원본영상에 마스크 mxn 행렬을 대조할경우 
마스크 값이 모두 1일 경우 계산된 영상엔 
중간값이 1 or 마스크가 2x2인경우 오른쪽부터 1
* 2. Dilation 팽창
Erosion과 반대로 마스크 값이 하나라도 1이면 
계산된 영상엔 중간값이 1
* Opening 침식 -> 팽창 
* Closing 팽창 -> 침식

* 4.decision moving object
* 움직임판단
* Moving Average
연속적인 프레임을 계속해서 평균을 내면서 배경을 업데이트 시키는 방법
* addWeighted(영상, alpha, BG영상,beta,0.0,BG영상);
 영상,BG영상 두 프레임이 여러가지의 비율 중 
 하날 사용해서 합을 BG영상이 서서히 변하는 함수(가중치함수)
*/

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#ifdef _DEBUG
#pragma comment(lib,"opencv_core453d.lib")  
#pragma comment(lib,"opencv_imgproc453d.lib")
#pragma comment(lib,"opencv_imgcodecs453d.lib")  
#pragma comment(lib,"opencv_dnn453d.lib")
#pragma comment(lib,"opencv_highgui453d.lib")
#pragma comment(lib,"opencv_videoio453d.lib")
#pragma comment(lib,"tbb12.lib")
#pragma comment(lib,"tbb12_debug.lib")
#endif

using namespace cv;
using namespace std;

void draw_rect(Mat& img, vector<Rect>& v_rect)
{
	//받아온 값을 rectangle을 그려주는 함수
	for (auto it : v_rect) {
		rectangle(img, it, CV_RGB(255, 0 , 0), 2);
	}
}

int main(int, char)
{
	int TH1 = 100;
	// 픽셀값 차이가 TH1값 이상일때만 검출된다

	int TH_WIDTH = 10;
	int TH_HEIGHT = 10;
	// 검출된 blob에 가로세로 크기가 얼마 이상일때 검출할 건지 구하는 변수
	int TH_AREA;
	// blob에 면적이 얼마 이상일때만 검출할 건지 구하는 변수

	float detection_ratio = 0.02;
	// TH를 계산하기위한 변수

	float detection_area = 0.001;
	
	int alpha = 0.9; //이값을 낮출수록 업데이트 속도가 느려짐
	int beta = (1.0 - alpha);
	int min_area_percent = 0.1;

	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5), Point(2, 2));

	Mat frame, frame_binary;
	Mat old_frame, bg_frame_binary;
	Mat sub_frame;
	//Mat absdiff_frame
	VideoCapture stream1("4K Road traffic video for object detection and tracking - free download now!.mp4"); 

	if (!stream1.isOpened()){cout << "Cannot open camera";}

	namedWindow("frame", 0);
	namedWindow("old_frame", 0);
	namedWindow("sub_frame", 0);
	namedWindow("bg_frame_binary", 0);

	int count = 0;
	int SKIP = 50;
	//50프레임 정도는 그냥 흘려보낸다 (처음에 프레임이 깨져서 들어올수도 있기때문)

	while (1)
	{
		if(!(stream1.read(frame))) // get one frame form video
			break;
		
		if (old_frame.empty())
		{
			old_frame = frame.clone(); //51프레임에서 BG가 들어오고
			cvtColor(old_frame, bg_frame_binary, COLOR_BGR2GRAY);

			//TH_AREA = bg_frame_binary.size().area() * min_area_percent / 100.0;

			//저장된 영상을 GRAY값으로 변환해준다
			TH_WIDTH = int(old_frame.cols * detection_ratio);
			TH_HEIGHT = int(old_frame.rows * detection_ratio);
			//배경의 가로(cols), 세로(rows) 크기의 몇퍼센트를 width,height TH로 사용할것인지 정하는함수 
			TH_AREA = int((old_frame.cols * old_frame.rows)* detection_area);
			//detection_area을 곱해줌으로써 TH_AREA가 v_rect 에다가 TH값을 넣어줌
			continue;
		}

		cvtColor(old_frame, frame_binary, COLOR_BGR2GRAY);
		addWeighted(frame_binary, alpha, bg_frame_binary, beta, 0.0, bg_frame_binary);

		if (count < SKIP) {
			count++;
			printf("skip : %d\n", count);
			continue;
		}

		//subtract(old_frame, frame, sub_frame);
		// old_frame과 frame이 빼기를 해서 sub_frame 에 넣는다
		// old_frame - frame = sub_frame
		absdiff(bg_frame_binary, frame_binary, sub_frame);
		threshold(sub_frame, sub_frame,TH1,255,THRESH_BINARY);
		morphologyEx(sub_frame, sub_frame, MORPH_CLOSE, element);
		//imshow("absdiff_frame", absdiff_frame);
		//sub_frame 이진화된 프레임
		
		//find contour
		vector< vector< Point> >contours;
		vector<Vec4i>hierarchy;
		findContours(sub_frame.clone(), contours, hierarchy,RETR_CCOMP,CHAIN_APPROX_SIMPLE);
		// findContour : blob을 찾는 함수
		// drawContours(frame, contours, -1, CV_RGB(255,0,0),5,7,hierarchy);
		
		//Blob labeling
		//blob이란 데이터를 포함할 수 있는 다차원의 데이터 표현 방식
		// (예외의 검출되는 것들,레티클)
		//OpenCV에서 blob은 4차원의 Mat으로 표현
		vector<Rect>v_rect;
		for (auto it : contours) //ranged-based for statement for(dtype dst:src)
		{
			//Rect mr = boundingRect(Mat(it));
			//v_rect.push_back(mr);

			//if (mr.width > TH_WIDTH || mr.height > TH_HEIGHT)
			//	v_rect.push_back(mr);

			//blob의 면적을 이용한 방법
			double area = contourArea(it, false);
			//auto 선언하는 인스턴스(변수)의 형식이 '자동'으로 결정

			if (area > TH_AREA)
			{
				Rect mr = boundingRect(Mat(it));
				v_rect.push_back(mr);
				//printf("%lf\n", area);
			}
		}
		draw_rect(frame, v_rect);

		imshow("frame", frame);
		imshow("old_frame", old_frame);
		imshow("sub_frame", sub_frame);
		imshow("bg_frame_binary", bg_frame_binary);

		old_frame = frame.clone();

		if (waitKey(5) >= 0)
			break;
	}

	/*
	//Morphology
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	Mat img;
	img = imread("C:/opencvproj/testimages/sam-moqadam-UkwbRZkt8zM-unsplash.jpg");
	resize(img, img, Size(640, 480));

	Mat element = getStructuringElement(MORPH_RECT, Size(3,3),Point(1,1));
	//MORPH_RECT는 3x3 에 1로 다 채운것이고
	//MORPH_ELLIPSE는 3x3 에 원같은 모양으로 1이 채워지고
	//MORPH_CROSS는 3x3에 십자가 형태로 1이 채워진다

	Mat rImg;
	morphologyEx(img, rImg, MORPH_OPEN, element);
	//morphologyEx(img, img, MORPH_CLOSE, element); -> It's also ok.
	//morphologyEx(src, dst,Opening or Closing , 마스크값);

	namedWindow("i");
	imshow("i", img);
	namedWindow("r");
	imshow("r", rImg);
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	waitKey(0);
	*/
	return 0;
	
	
}
