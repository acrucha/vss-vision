#include "ImageProcessing.h"


cv::Mat EdgesSegmentation::low_contrast(cv::Mat& img) {
    const double ALPHA_OFFSET = 0.8;

    double min_val, max_val;
    cv::minMaxLoc(img, &min_val, &max_val);

    double alpha = 255 / (max_val - min_val);
    double beta = -alpha * min_val;

    cv::Mat img_contrast;
    cv::convertScaleAbs(img, img_contrast, alpha * ALPHA_OFFSET, beta);

    // increase saturation
    cv::cvtColor(img_contrast, img_contrast, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;
    cv::split(img_contrast, channels);
    // channels[0] = channels[1] * 1.2; //hue
    channels[1] = channels[1] * 0.6; 
    cv::merge(channels, img_contrast);
    cv::cvtColor(img_contrast, img_contrast, cv::COLOR_HSV2BGR);

    return img_contrast;
}

bool EdgesSegmentation::not_on_the_border(int height, int width, const std::vector<cv::Point>& contour) {
    // Obtém o retângulo delimitador do contorno
    cv::Rect bounding_rect = cv::boundingRect(contour);

    // Verifica se o retângulo delimitador não coincide com as bordas da imagem
    return (bounding_rect.x > 0 && bounding_rect.y > 0 &&
            bounding_rect.x + bounding_rect.width < width &&
            bounding_rect.y + bounding_rect.height < height);
}

cv::Mat EdgesSegmentation::polygon_contours(cv::Mat& img_proc, cv::Mat& img_base) {
    const int thresh = 80, max_thresh = 255; 
    const int height = img_proc.rows, width = img_proc.cols;
    const int MAX_RADIUS = 24, MIN_RADIUS = 15;

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(img_proc, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    std::vector<std::vector<cv::Point>> contours_poly(contours.size());
    std::vector<cv::Rect> boundRect(contours.size());
    std::vector<cv::Point2f> center(contours.size());
    std::vector<float> radius(contours.size());

    for(int i = 0; i < contours.size(); i++){
        cv::approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true);
        boundRect[i] = cv::boundingRect( cv::Mat(contours_poly[i]));
        cv::minEnclosingCircle((cv::Mat)contours_poly[i], center[i], radius[i]);
    }

    for(int i = 0; i< contours.size(); i++){
        if(not_on_the_border(height, width, contours[i]) 
            && radius[i] <= MAX_RADIUS && radius[i] >= MIN_RADIUS 
            && std::abs(boundRect[i].height - boundRect[i].width) < MAX_RADIUS){

            cv::Scalar RED = cv::Scalar(0, 0, 255);
            cv::Scalar GREEN = cv::Scalar(0, 255, 0);
            cv::drawContours(img_base, contours_poly, i, RED, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
            cv::rectangle(img_base, boundRect[i].tl(), boundRect[i].br(), RED, 2, 8, 0);
            cv::circle(img_base, center[i], (int) radius[i], RED, 2, 8, 0);
        }
    }

    return img_base;
}

cv::Mat EdgesSegmentation::softLightFocus(cv::Mat& img_base){
    cv::Mat smoothed_image;
    cv::bilateralFilter(img_base, smoothed_image, 7, 20, 5);
    return smoothed_image;
}

cv::Mat EdgesSegmentation::sharpnessIncrease(cv::Mat& img_base){
    cv::Mat f_enhanced;
	cv::detailEnhance(img_base, f_enhanced, 15, 0.4);

    cv::Mat sharpened_image;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    cv::filter2D(f_enhanced, sharpened_image, -1, kernel);

    return sharpened_image;
}

cv::Mat EdgesSegmentation::thresholding(cv::Mat& img_base){
    cv::Mat gray, thresh, adptThresh;
    cv::cvtColor(img_base, gray, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(gray, gray, cv::Size(9, 9), 0);
    cv::threshold(gray, thresh, 200, 255, cv::THRESH_BINARY+cv::THRESH_OTSU);

    return thresh;
}

cv::Mat EdgesSegmentation::run(cv::Mat& img_base) {

    cv::Mat img;
    img = low_contrast(img_base);

    img = softLightFocus(img);

    img = sharpnessIncrease(img);

    img = thresholding(img);

    img = polygon_contours(img, img_base);

    return img;
}

