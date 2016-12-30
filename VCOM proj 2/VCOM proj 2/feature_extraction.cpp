
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <sstream>
#include <fstream>


using namespace std;
using namespace cv;

const  int AIRPLANE = 0;
const  int AUTOMOBILE = 1;
const  int BIRD = 2;
const  int CAT = 3;
const  int DEER = 4;
const  int DOG = 5;
const  int FROG = 6;
const  int HORSE = 7;
const  int SHIP = 8;
const  int TRUCK = 9;

int num_clusters = 10;
Mat labels(0, 10, CV_32FC1);
Mat trainingData(0, num_clusters, CV_32FC1);
Mat input;
SiftFeatureDetector detector;
vector<int> imgLabels;

void csvToArray() {
	ifstream file("C:\\Users\\diogo\\Downloads\\trainLabels.csv");

	for (int row = 1; row < 50001; row++)
	{
		string line;
		getline(file, line);
		if (!file.good())
			break;

		
			
			string val;
			string delimiter = ",";
			string val = line.substr(line.find(delimiter)+1, line.length);
			int label;
			if (val == "airplane")
				label = AIRPLANE;
			else if (val == "automobile")
				label = AUTOMOBILE;
			else if (val == "bird")
				label = BIRD;
			else if (val == "cat")
				label = CAT;
			else if (val == "deer")
				label = DEER;
			else if (val == "dog")
				label = DOG;
			else if (val == "frog")
				label = FROG;
			else if (val == "horse")
				label = HORSE;
			else if (val == "ship")
				label = SHIP;
			else label = TRUCK;
			imgLabels.push_back(label);
		
	}
	}

void imageTrainer(BOWImgDescriptorExtractor bowDE) {
	std::vector<cv::KeyPoint> keypoints;
		Mat bowDescriptor;
	for (int i = 1; i <= 50000; i++) {
		String path = "C:\\Users\\diogo\\Downloads\\train\\" + to_string(i) + ".png";
		input = imread(path, 0); //Load as grayscale

		cout << i << endl;
		detector.detect(input, keypoints);
		bowDE.compute(input, keypoints, bowDescriptor);
		trainingData.push_back(bowDescriptor);

		labels.push_back(imgLabels[i-1]);
	}
}

int train()
{

	
	std::vector<cv::KeyPoint> keypoints;
	Mat keypointImg;		
	DescriptorExtractor* descriptorExtractor;
	Mat descriptors, allDescriptors;// 
	for (int i = 1; i < 20; i++) {
	//START sift detector
		String path = "C:\\Users\\diogo\\Downloads\\test\\" + to_string(i) + ".png";
		input = imread(path, 0); //Load as grayscale
		detector.detect(input, keypoints);

	// Add results to image and show.
	
	drawKeypoints(input, keypoints, keypointImg);
	//imshow("keypoints", keypointImg);

	
	
	descriptorExtractor = new SiftDescriptorExtractor();
	descriptorExtractor->compute(input, keypoints, descriptors);
	allDescriptors.push_back(descriptors);


}
	//kmeans clustering
	BOWKMeansTrainer bow(num_clusters);
	bow.add(allDescriptors);
	Mat vocabulary = bow.cluster();
	//save vocabulary to file
	FileStorage fs1("vocabulary.yml", FileStorage::WRITE);
	fs1 << "vocabulary" << vocabulary;
	fs1.release();

	descriptorExtractor = new SiftDescriptorExtractor();
	FlannBasedMatcher* matcher = new FlannBasedMatcher();
	BOWImgDescriptorExtractor bowDescriptorExtractor(descriptorExtractor , matcher);
	bowDescriptorExtractor.setVocabulary(vocabulary);

	
	// Set up SVM's parameters
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	CvSVM SVM;
	

	// Train the SVM
	vector<KeyPoint> keypoint;
	Mat bowDescriptor;

	

	printf("%s\n", "Training SVM classifier");

	SVM.train(trainingData, labels, Mat(), Mat(), params);
	SVM.save("test.xml");
	
	
	
	
	return 1;

}