#include "hog_svm.h"
#include <opencv2/opencv.hpp>
#include "tools.h"
using namespace cv;
using namespace std;
using namespace cv::ml;


static void CoumputeHog(const Mat& src, vector<float> &descriptors)
{
    HOGDescriptor myHog = HOGDescriptor(IMAGE_SIZE, Size(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
    myHog.compute(src.clone(), descriptors, Size(1, 1), Size(0, 0));

}
static int GetSamples(const string & samplepath ,const int lable, vector<Mat> * vecImages, vector<int> *  vecLabels)
{
    vector<string> files;
    files = tools::FilenamesInDir(samplepath);

    for (int i = 0; i < files.size(); i++)
    {
        if (files[i].find("jpg") != std::string::npos || files[i].find("png") != std::string::npos)
        {
            string imgpath = samplepath + files[i];
            cout << "read:" << imgpath << endl << flush;
            Mat img = imread(imgpath);
            //pyrUp(img,img);

            if (img.empty())
            {
                fprintf(stderr, "failed to open \n");
                return -1;
            }
            resize(img, img, IMAGE_SIZE);
            vecImages->push_back(img);
            vecLabels->push_back(lable);
        }
    }
}
HogSVM::HogSVM()
{
    Train();
}
HogSVM::~HogSVM()
{
    
}
int HogSVM::Train()
{
    string imageName;
    signed imageLabel;
    vector<Mat> vecImages;
    vector<int> vecLabels;
    vector<int> predict;
    svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));
    vector<float> vecDescriptors;
    string bracketsamplepath("D:\\SVMS\\brackets\\");
    string othersamplepath("D:\\SVMS\\others\\");
    GetSamples(bracketsamplepath, 0, &vecImages, &vecLabels);
    GetSamples(othersamplepath, 1, &vecImages, &vecLabels);


    Mat dataDescriptors;
    Mat dataResponse = (Mat)vecLabels;
    for (size_t i = 0; i < vecImages.size(); i++)
    {
        Mat src = vecImages[i];
        Mat tempRow;
        CoumputeHog(src, vecDescriptors);
        if (i == 0)
        {
            dataDescriptors = Mat::zeros(vecImages.size(), vecDescriptors.size(), CV_32FC1);
        }
        tempRow = ((Mat)vecDescriptors).t();
        tempRow.row(0).copyTo(dataDescriptors.row(i));
    }
    svm->train(dataDescriptors, ROW_SAMPLE, dataResponse);
    return 0;
}


int HogSVM::Predict_(const Mat & img) const
{
    if (!tools::IsImgLegal(img))
    {
        return -1;
    }
    Mat test = img;
    resize(test, test, IMAGE_SIZE);
    vector<float> imageDescriptor;
    CoumputeHog(test, imageDescriptor);
    Mat testDescriptor = Mat::zeros(1, imageDescriptor.size(), CV_32FC1);
    for (size_t i = 0; i < imageDescriptor.size(); i++)
    {
        testDescriptor.at<float>(0, i) = imageDescriptor[i];
    }
    float  label = svm->predict(testDescriptor);
    /*
    cout << label << endl;
    imshow("test image", test);
    waitKey(0);
    */
    return label;
}


int main(int argc, char** argv) 
{
 
    HogSVM hogsvm;

    string         dir("D:\\images\\");
    vector<string> files;
    files = tools::FilenamesInDir(dir);

    for (int i = 0; i < files.size(); i++)
    {
        if (files[i].find("jpg") != std::string::npos || files[i].find("png") != std::string::npos)
        {
            string imgpath = dir + files[i];
            Mat img = imread(imgpath);
            //pyrUp(img,img);

            if (img.empty())
            {
                fprintf(stderr, "failed to open \n");
                return -1;
            }
            Mat imgbin;
            tools::AdapBinImg(img, imgbin);
            vector<Rect> boxes;
            tools::CCl(imgbin, boxes, 0);
            int index = 0;
            for (auto box : boxes)
            {
                int lab = hogsvm.Predict_(imgbin(box));
                if (lab != -1)
                {
                    putText(img, to_string(lab), box.tl(), 1, 1, Scalar(0, 0, 0));
                }
            }
            imshow("img", img);
            waitKey(0);
        }
        
    }

    return 0;
}