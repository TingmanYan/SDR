/*********************************************************************
 * This code is copyed from taniai
 * Functions to evaluate error of disparity maps
 ********************************************************************/
#pragma once

#include "TimeStamper.h"
#include <opencv2/opencv.hpp>

class Evaluator
{

protected:
	TimeStamper timer;
	const cv::Mat dispGT;
	const cv::Mat nonoccMask;
	cv::Mat occMask;
	cv::Mat validMask;
	std::string saveDir;
	std::string header;
	int validPixels;
	int nonoccPixels;
	FILE *fp_output;
	double errorThreshold;

public:
	bool showProgress;
	bool saveProgress;
	bool printProgress;

	std::string getSaveDirectory()
	{
		return saveDir;
	}

	Evaluator(cv::Mat dispGT, cv::Mat nonoccMask, std::string header = "result", std::string saveDir = "./", bool show = true, bool print = true, bool save = true)
		: dispGT(dispGT)
		, nonoccMask(nonoccMask)
		, saveDir(saveDir)
		, header(header)
		, fp_output(nullptr)
		, showProgress(show)
		, saveProgress(save)
		, printProgress(print)
	{

		if (save)
		{
			fp_output = fopen((saveDir + "log_output.txt").c_str(), "w");
			if (fp_output != nullptr)
			{
				fprintf(fp_output, "%s\t%s\t%s\n", "Time", "all", "nonocc");
				fflush(fp_output);
			}
		}

		errorThreshold = 0.5;

		validMask = (dispGT > 0.0) & (dispGT != INFINITY);
		validPixels = cv::countNonZero(validMask);
		occMask = ~nonoccMask & validMask;
		nonoccPixels = cv::countNonZero(nonoccMask);
	}
	~Evaluator()
	{
		if (fp_output != nullptr) fclose(fp_output);
	}


	void setErrorThreshold(double t)
	{
		errorThreshold = t;
	}

	void evaluate(cv::Mat disp, bool save, bool print)
	{
		cv::Mat errorMap = cv::abs(disp - dispGT) <= errorThreshold;
		cv::Mat errorMapVis = errorMap | (~validMask);
		//errorMapVis.setTo(cv::Scalar(200), occMask & (~errorMapVis));

		double all = 1.0 - (double)cv::countNonZero(errorMap & validMask) / validPixels;
		double nonocc = 1.0 - (double)cv::countNonZero(errorMap & nonoccMask) / nonoccPixels;
		all *= 100.0;
		nonocc *= 100.0;

		if (saveProgress && save) {
			cv::imwrite(saveDir + cv::format("%sE.png", header.c_str()), errorMapVis);

			if (fp_output != nullptr)
			{
				fprintf(fp_output, "%lf\t%lf\t%lf\n", getCurrentTime(), all, nonocc);
				fflush(fp_output);
			}
		}

		if (printProgress && print) 
			std::cout << cv::format("%5.1lf\t%4.2lf\t%4.2lf", getCurrentTime(), all, nonocc) << std::endl;
	}

	void start()
	{
		timer.start();
	}

	void stop()
	{
		timer.stop();
	}

	double getCurrentTime()
	{
		return timer.getCurrentTime();
	}

};
