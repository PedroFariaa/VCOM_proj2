#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>


using namespace std;
using namespace cv;


int main(int, char** argv)
{

	Mat input;
	SiftFeatureDetector detector;
	std::vector<cv::KeyPoint> keypoints;
	Mat keypointImg;		
	DescriptorExtractor* descriptorExtractor;
	Mat descriptors, allDescriptors;// 
	for (int i = 1; i < 20; i++) {
	//START sift detector
		int a = 10;
		stringstream ss;
		ss << a;
		string str = ss.str();
		String path = "C:\\Users\\diogo\\Downloads\\test\\" + str + ".png";
		input = imread(path, 0); //Load as grayscale
		detector.detect(input, keypoints);

	// Add results to image and show.
	
	drawKeypoints(input, keypoints, keypointImg);
	//imshow("keypoints", keypointImg);

	//kmeans clustering
	
	descriptorExtractor = new SiftDescriptorExtractor();
	descriptorExtractor->compute(input, keypoints, descriptors);
	allDescriptors.push_back(descriptors);


}
	BOWKMeansTrainer bow(10);
	bow.add(allDescriptors);
	Mat vocabulary = bow.cluster();
	return 0;

}