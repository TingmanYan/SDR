#include <sys/stat.h>
#include <chrono>
#include "ArgsParser.h"
#include "Evaluator.h"
#include "FastDR.h"
#include "Utilities.hpp"

#ifdef _WIN32
#include <direct.h>
#endif

/**
 * Winner-take-all operation to compute disparity
 */
cv::Mat WTAOfVolume(const cv::Mat& volume)
{
	int D = volume.size.p[0];
	int H = volume.size.p[1];
	int W = volume.size.p[2];
	cv::Mat disp(H, W, CV_32F), mask;
	cv::Mat cost(H, W, CV_32F);
	cv::Mat cost_minimum(H, W, CV_32F, cv::Scalar::all(FLT_MAX));
	cv::Mat disp_WTA(H, W, CV_32F, cv::Scalar::all(0));
	for (int d = 0; d < D; d++) {
		disp = d;
		for (int y = 0; y < H; y++) {
			auto p = volume.ptr<float>(d, y);
			for (int x = 0; x < W; x++) {
				cost.at<float>(y, x) = *(p + x);
			}
		}
		mask = cost < cost_minimum;
		cost.copyTo(cost_minimum, mask);
		disp.copyTo(disp_WTA, mask);
	}
	return disp_WTA;
}

/**
 * Filling the left (right) part of the cost volume
 */
void fillOutOfView(cv::Mat& volume)
{
	int D = volume.size.p[0];
	int H = volume.size.p[1];
	int W = volume.size.p[2];

	for (int d = 0; d < D; d++)
		for (int y = 0; y < H; y++)
		{
			auto p = volume.ptr<float>(d, y);
			auto q = p + d;
			float v = *q;
			while (p != q) {
				*p = v;
				p++;
			}
		}
}

/**
 * Loading the data of Middlebury dataset
 */
bool loadData(const std::string inputDir, cv::Mat& im0, cv::Mat& disp_WTA, cv::Mat& dispGT, cv::Mat& nonocc, Calib&calib)
{
	if (calib.ndisp <= 0)
		printf("Try to retrieve ndisp from file [calib.txt].\n");
	calib = Calib(inputDir + "calib.txt");
	if (calib.ndisp <= 0) {
		printf("ndisp is not speficied.\n");
		return false;
	}

	im0 = cv::imread(inputDir + "im0.png");
	if (im0.empty()) {
		printf("Image im0.png not found in\n");
		printf("%s\n", inputDir.c_str());
		return false;
	}

	disp_WTA = cvutils::io::read_pfm_file(inputDir + "disp_WTA.pfm");
	if (disp_WTA.empty()) {
		int sizes[] = { calib.ndisp, im0.rows, im0.cols };
		cv::Mat vol = cv::Mat_<float>(3, sizes);
		if (cvutils::io::loadMatBinary(inputDir + "im0.acrt", vol, false) == false) {
			printf("Cost volume file im0.acrt not found\n");
			return false;
		}
		fillOutOfView(vol);
		disp_WTA = WTAOfVolume(vol);
		cvutils::io::save_pfm_file(inputDir + "disp_WTA.pfm", disp_WTA);
	}

	dispGT = cvutils::io::read_pfm_file(inputDir + "disp0GT.pfm");
	if (dispGT.empty())
		dispGT = cv::Mat_<float>::zeros(im0.size());

	nonocc = cv::imread(inputDir + "mask0nocc.png", cv::IMREAD_GRAYSCALE);
	if (!nonocc.empty())
		nonocc = nonocc == 255;
	else
		nonocc = cv::Mat_<uchar>(im0.size(), 255);

	return true;
}

/**
 * Processing Middlebury
 */
void MidV3(const std::string inputDir, const std::string outputDir, const Options& options)
{
    Parameters params = options.params;

	cv::Mat im0, disp_WTA, dispGT, nonocc;
	Calib calib;

	calib.ndisp = options.ndisp;
        if (!loadData(inputDir, im0, disp_WTA, dispGT, nonocc, calib))
            return;
	printf("ndisp = %d\n", calib.ndisp);
	
	int maxdisp = calib.ndisp;
	double errorThresh = 1.0;
	if (cvutils::contains(inputDir, "trainingQ") || cvutils::contains(inputDir, "testQ"))
		errorThresh = errorThresh / 2.0;
	else if (cvutils::contains(inputDir, "trainingF") || cvutils::contains(inputDir, "testF"))
		errorThresh = errorThresh * 2.0;

	{
#ifdef _WIN32
        _mkdir((outputDir + "debug").c_str());
#elif defined __linux__ || defined __APPLE__
        mkdir((outputDir + "debug").c_str(), 0755);
#endif

        Evaluator* eval =
            new Evaluator(dispGT, nonocc, "result", outputDir + "debug/");
        eval->setErrorThreshold(errorThresh);
        eval->start();

        FastDR fdr(im0, disp_WTA, params, maxdisp, 0);

        cv::Mat labeling, refined_disp;
        fdr.run(labeling, refined_disp);

        cvutils::io::save_pfm_file(outputDir + "disp0FDR.pfm",
                                   refined_disp);

        {
            FILE* fp = fopen((outputDir + "timeFDR.txt").c_str(), "w");
            if (fp != nullptr) {
                fprintf(fp, "%lf\n", eval->getCurrentTime());
                fclose(fp);
            }
        }
        if (cvutils::contains(inputDir, "training"))
            eval->evaluate(refined_disp, true, true);

        delete eval;
    }
}

/**
 * Update the invalid disparity (0 value) to a random disparity from
 * (1,max_disp)
 */
void preprocess_disp(cv::Mat& disp_WTA, const int max_disp) {
	cv::RNG rng;
	for (int y = 0; y < disp_WTA.rows; y++)
		for (int x = 0; x < disp_WTA.cols; x++) {
			if (disp_WTA.at<float>(y, x) < 1)
				disp_WTA.at<float>(y, x) = rng.operator() (max_disp - 1) + 1;
		}
}

/**
 * Processing KITTI
 */
void KITTI(const std::string inputDir, const std::string outputDir,
	const Options& options) {
    Parameters params = options.params;

    int max_disp = options.ndisp;

    cv::Mat im0, disp_WTA_u16, disp_WTA;

    auto c0 = std::chrono::steady_clock::now();

    for (int i = 0; i < 200; i++) {
        std::cout << "--------------------------------------------"
                  << std::endl;
        std::cout << "processing: " + cv::format("%06d_10.png", i) << std::endl;

        im0 = cv::imread(inputDir + "image_2/" + cv::format("%06d_10.png", i));
        if (im0.empty()) {
            printf("Color reference image not found in\n");
            printf("%s\n", inputDir.c_str());
            return;
        };
        disp_WTA_u16 =
            cv::imread(inputDir + "disp_WTA/" + cv::format("%06d_10.png", i),
                       CV_LOAD_IMAGE_UNCHANGED);
        if (disp_WTA_u16.empty()) {
            printf("WTA disparity map not found in\n");
            printf("%s\n", inputDir.c_str());
            return;
        };

        // disp_WTA_u16.convertTo(disp_WTA, CV_32FC1, 1/256.0);
        disp_WTA_u16.convertTo(disp_WTA, CV_32FC1);
        preprocess_disp(disp_WTA, max_disp);

        FastDR* fdr = new FastDR(im0, disp_WTA, params, max_disp, 0);

        cv::Mat labeling, refined_disp;
        fdr->run(labeling, refined_disp);

        refined_disp.convertTo(disp_WTA_u16, CV_16UC1, 256);

        cv::imwrite(outputDir + cv::format("%06d_10.png", i), disp_WTA_u16);

        delete fdr;
    }

    auto c1 = std::chrono::steady_clock::now();
    std::cout << "Total time consumed: "
              << std::chrono::duration_cast<std::chrono::microseconds>(c1 - c0) .count()
              << std::endl;
}

int main(int argc, const char** args) {
    std::cout << "----------- parameter settings -----------" << std::endl;
    ArgsParser parser(argc, args);
    Options options;

    options.loadOptionValues(parser);

    if (options.outputDir.length()) {
#ifdef _WIN32
        _mkdir((options.outputDir).c_str());
#elif defined __linux__ || defined __APPLE__
        mkdir((options.outputDir).c_str(), 0755);
#endif
    }

    if (options.mode == "MiddV3") {
        printf("Running by Middlebury V3 mode.\n");
        printf(
            "This mode assumes MC-CNN matching cost files im0.acrt or WTA "
            "disparities of the cost disp_WTA.pfm in targetDir.\n");
        MidV3(options.targetDir + "/", options.outputDir + "/", options);
    } else if (options.mode == "KITTI") {
        printf("Running by KITTI mode.\n");
        printf(
            "This mode assumes pre-computed WTA disparity maps in "
            "targetDir.\n");
        KITTI(options.targetDir + "/", options.outputDir + "/", options);
    } else {
        printf("Specify the following arguments:\n");
        printf("  -mode [MiddV3, KITTI]\n");
        printf("  -targetDir [PATH_TO_IMAGE_DIR]\n");
        printf("  -outputDir [PATH_TO_OUTPUT_DIR]\n");
    }

    return 0;
}
