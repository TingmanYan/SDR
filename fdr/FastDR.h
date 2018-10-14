#pragma once

#include <opencv2/opencv.hpp>
#include "SurfaceFitting.h"
#include <time.h>

class FastDR {
   protected:
    const int width;
    const int height;
    const int MAX_DISPARITY;
    const int MIN_DISPARITY;
    cv::Mat I;
    cv::Mat D;

    cv::Mat spx;
    int num_spx;
    uchar **adj_mat = NULL;

    SpxFeature *sfeature = NULL;

   public:
    Parameters params;

    FastDR(cv::Mat im0, cv::Mat disp_WTA, Parameters params, int maxDisparity,
           int miniDisparity = 0)
        : width(im0.cols),
          height(im0.rows),
          MAX_DISPARITY(maxDisparity),
          MIN_DISPARITY(miniDisparity),
          params(params) {
        I = im0.clone();

        D = disp_WTA.clone();

        clock_t start, end;
        start = clock();

        spx = cv::Mat(im0.size(), CV_32SC1);
        initSuperpixels(I, spx, num_spx);

        end = clock();
        std::cout << "Time consumed in Segmentation: "
                  << (double)(end - start) / CLOCKS_PER_SEC << std::endl;

        adj_mat = new uchar *[num_spx];
        for (int k = 0; k < num_spx; k++)
            adj_mat[k] = new uchar[num_spx]{0};

        sfeature = new SpxFeature[num_spx];
    }

    ~FastDR() {
        for (int k = 0; k < num_spx; k++)
            if (adj_mat[k]) delete[] adj_mat[k];
        if (adj_mat) delete[] adj_mat;
        if (sfeature) delete[] sfeature;
    }

    void run(cv::Mat &labeling, cv::Mat &refined_disp) {
        clock_t start, end;
        start = clock();

        computeSpxFeature(I, spx, num_spx, sfeature);

        MRFInference(D, spx, num_spx, adj_mat, sfeature, params);

        end = clock();
        std::cout << "Time consumed in MRF inference: "
                  << (double)(end - start) / CLOCKS_PER_SEC << std::endl;

        SurfaceFitting sfitting(D, sfeature, spx, num_spx, adj_mat, MAX_DISPARITY, params);
        sfitting.process(labeling, refined_disp);

        adaptiveMeanFilter(refined_disp);
        medianFilter(refined_disp);
    }

   protected:
    void initSuperpixels(const cv::Mat I, cv::Mat &spx, int &num_spx) {
        num_spx = 0;
        graphSegmentation(I, spx, num_spx);
        CV_Assert(num_spx != 0);
        std::cout << "Number of superpixels: " << num_spx << std::endl;
    }

    void graphSegmentation(const cv::Mat I, cv::Mat &spx, int &num_spx);

    void computeSpxFeature(const cv::Mat I, const cv::Mat spx,
                           const int num_spx, SpxFeature *sfeature);

    void MRFInference(const cv::Mat D, const cv::Mat spx, const int num_spx,
                      uchar **adj_mat, SpxFeature *sfeature,
                      const Parameters params) {
        int **edge_length = new int *[num_spx];
        for (int k = 0; k < num_spx; k++)
            edge_length[k] = new int[num_spx]{0};
        computeAdjmat(spx, adj_mat, edge_length);

        const int num_bins = (MAX_DISPARITY - 1) / params.bins_width + 1;
        float **data_term = new float *[num_spx];
        for (int k = 0; k < num_spx; k++)
            data_term[k] = new float[num_bins]{0};
        computeDataTerm(D, spx, num_spx, sfeature, data_term, params.bins_width);

        graphCutOptimization(data_term, adj_mat, edge_length, sfeature,
                             num_bins, num_spx, params);

        for (int k = 0; k < num_spx; k++)
            delete[] edge_length[k];
        delete[] edge_length;
        for (int k = 0; k < num_spx; k++)
            delete[] data_term[k];
        delete[] data_term;
    }

    void computeAdjmat(const cv::Mat spx, uchar **adj_mat, int **edge_length);

    void computeDataTerm(const cv::Mat D, const cv::Mat spx, const int num_spx,
                         SpxFeature *sfeature, float **data_term,
                         const int bins_width);

    void graphCutOptimization(float **data_term, uchar **adj_mat,
                              int **edge_length, SpxFeature *sfeature,
                              const int num_bins, const int num_spx,
                              const Parameters params);

    void adaptiveMeanFilter(cv::Mat &D);

    void medianFilter(cv::Mat &D);
};
