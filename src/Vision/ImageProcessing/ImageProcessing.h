#ifndef IMAGEPROCESSING_H
#define IMAGEPROCESSING_H

#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <mutex>

/**
 * @brief    Class for image processing.
 */
class ImageProcessing {
public:
	/**
	 * @brief    run algorithm
	 *
	 * @param[in]  frame  The frame
	 *
	 * @return   frame with the filter
	 */
  virtual cv::Mat run(cv::Mat& frame) = 0;

	/**
	 * @brief    Gets the debug frame.
	 *
     * @param[in]  frame  The debug frame
	 */
  virtual void getDebugFrame(cv::Mat& frame) = 0;

  virtual ~ImageProcessing() = default;
	
};

namespace EdgesSegmentation {
	cv::Mat low_contrast(cv::Mat& img);
	cv::Mat polygon_contours(cv::Mat& img_proc, cv::Mat& img_base);
	bool not_on_the_border(int height, int width, const std::vector<cv::Point>& contour);
	cv::Mat run(cv::Mat& img);
	cv::Mat softLightFocus(cv::Mat& img_base);
	cv::Mat sharpnessIncrease(cv::Mat& img_base);
	cv::Mat thresholding(cv::Mat& img_base);
}

#endif
