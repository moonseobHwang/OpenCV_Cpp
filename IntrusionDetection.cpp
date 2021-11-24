/*
* Intrusion Detection Project
* 
* 1.input image
* �Է¿����� ���念��,ķ���κ��� �޾ƿ���
* 2.BG Subtraction
* ��� ����
* 3.noise remove
* ���� ����
* convolution
* 1. Erosion ħ��
�������� ����ũ mxn ����� �����Ұ�� 
����ũ ���� ��� 1�� ��� ���� ���� 
�߰����� 1 or ����ũ�� 2x2�ΰ�� �����ʺ��� 1
* 2. Dilation ��â
Erosion�� �ݴ�� ����ũ ���� �ϳ��� 1�̸� 
���� ���� �߰����� 1
* Morphology 
* Opening ħ�� -> ��â 
* Closing ��â -> ħ��

* 4.decision moving object
* �������Ǵ�
* Moving Average
�������� �������� ����ؼ� ����� ���鼭 ����� ������Ʈ ��Ű�� ���
* addWeighted(����, alpha, BG����,beta,0.0,BG����);
 ����,BG���� �� �������� ���������� ���� �� 
 �ϳ� ����ؼ� ���� BG������ ������ ���ϴ� �Լ�(����ġ�Լ�)

* 5.Upload Image to GoogleDrive
* ���� ����̺꿡 ���ε�

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
	//�޾ƿ� ���� rectangle�� �׷��ִ� �Լ�
	for (auto it : v_rect) {
		rectangle(img, it, CV_RGB(255, 0 , 0), 2);
	}
}

int main(int, char)
{
	int TH1 = 100;
	// �ȼ��� ���̰� TH1�� �̻��϶��� ����ȴ�

	int TH_WIDTH = 10;
	int TH_HEIGHT = 10;
	// ����� blob�� ���μ��� ũ�Ⱑ �� �̻��϶� ������ ���� ���ϴ� ����
	int TH_AREA;
	// blob�� ������ �� �̻��϶��� ������ ���� ���ϴ� ����

	float detection_ratio = 0.02;
	// TH�� ����ϱ����� ����

	float detection_area = 0.001;
	
	int alpha = 0.9; //�̰��� ������� ������Ʈ �ӵ��� ������
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
	//50������ ������ �׳� ��������� (ó���� �������� ������ ���ü��� �ֱ⶧��)

	while (1)
	{
		if(!(stream1.read(frame))) // get one frame form video
			break;
		
		if (old_frame.empty())
		{
			old_frame = frame.clone(); //51�����ӿ��� BG�� ������
			cvtColor(old_frame, bg_frame_binary, COLOR_BGR2GRAY);

			//TH_AREA = bg_frame_binary.size().area() * min_area_percent / 100.0;

			//����� ������ GRAY������ ��ȯ���ش�
			TH_WIDTH = int(old_frame.cols * detection_ratio);
			TH_HEIGHT = int(old_frame.rows * detection_ratio);
			//����� ����(cols), ����(rows) ũ���� ���ۼ�Ʈ�� width,height TH�� ����Ұ����� ���ϴ��Լ� 
			TH_AREA = int((old_frame.cols * old_frame.rows)* detection_area);
			//detection_area�� ���������ν� TH_AREA�� v_rect ���ٰ� TH���� �־���
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
		// old_frame�� frame�� ���⸦ �ؼ� sub_frame �� �ִ´�
		// old_frame - frame = sub_frame
		absdiff(bg_frame_binary, frame_binary, sub_frame);
		threshold(sub_frame, sub_frame,TH1,255,THRESH_BINARY);
		morphologyEx(sub_frame, sub_frame, MORPH_CLOSE, element);
		//imshow("absdiff_frame", absdiff_frame);
		//sub_frame ����ȭ�� ������
		
		//find contour
		vector< vector< Point> >contours;
		vector<Vec4i>hierarchy;
		findContours(sub_frame.clone(), contours, hierarchy,RETR_CCOMP,CHAIN_APPROX_SIMPLE);
		// findContour : blob�� ã�� �Լ�
		// drawContours(frame, contours, -1, CV_RGB(255,0,0),5,7,hierarchy);
		
		//Blob labeling
		//blob�̶� �����͸� ������ �� �ִ� �������� ������ ǥ�� ���
		// (������ ����Ǵ� �͵�,��ƼŬ)
		//OpenCV���� blob�� 4������ Mat���� ǥ��
		vector<Rect>v_rect;
		for (auto it : contours) //ranged-based for statement for(dtype dst:src)
		{
			//Rect mr = boundingRect(Mat(it));
			//v_rect.push_back(mr);

			//if (mr.width > TH_WIDTH || mr.height > TH_HEIGHT)
			//	v_rect.push_back(mr);

			//blob�� ������ �̿��� ���
			double area = contourArea(it, false);
			//auto �����ϴ� �ν��Ͻ�(����)�� ������ '�ڵ�'���� ����

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
	//MORPH_RECT�� 3x3 �� 1�� �� ä����̰�
	//MORPH_ELLIPSE�� 3x3 �� ������ ������� 1�� ä������
	//MORPH_CROSS�� 3x3�� ���ڰ� ���·� 1�� ä������

	Mat rImg;
	morphologyEx(img, rImg, MORPH_OPEN, element);
	//morphologyEx(img, img, MORPH_CLOSE, element); -> It's also ok.
	//morphologyEx(src, dst,Opening or Closing , ����ũ��);

	namedWindow("i");
	imshow("i", img);
	namedWindow("r");
	imshow("r", rImg);
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	waitKey(0);
	*/
	return 0;
	
	
}