#pragma once
#ifndef HOG_SVM_H
#define HOG_SVM_H
#include <opencv2/opencv.hpp>

const cv::Size IMAGE_SIZE = cv::Size(32, 32);
class HogSVM{

public:
    HogSVM();
    ~HogSVM();

    int Predict_(const cv::Mat & img) const;
    int Train();

private:
    cv::Ptr<cv::ml::SVM> svm;
};
#endif
