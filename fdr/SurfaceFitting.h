#pragma once
#include <opencv2/opencv.hpp>
#include "Parameters.h"
#include <time.h>

struct SpxFeature {
    cv::Vec3f color = cv::Vec3f(0, 0, 0);
    cv::Point2i tl;  // top left
    cv::Point2i br;  // bottom right
    int num_pixel = 0;
    int mean;
    double disp;
    bool occluded = false;
};

struct RgnFeature : SpxFeature {
    double ub;  // upper bound of disparity range
    double lb;  // lower bound of disparity range
    double disp_mean;
    double disp_var;
    double pd_weight = 0;  // posterior predictive probability

    int group_num;  // number of groups
    double inlier_rate = 0;
};

class SurfaceFitting {
   public:
    struct Group {
        int points_num = 0;
        int disp_num = 0;
        cv::Point3i* points = NULL;
        ~Group() {
            if (points) delete[] points;
        }
    };
    struct Configuration {
        int group_num;
        int points_num;
        int disp_num;
        int group_id[3];
        double sample_num;
        bool operator<(Configuration const& c) const;
    };

    struct Neighbor {
        int id;
        cv::Vec3d pl;  // plane label
        double prior_weight;  // prior
        double test_weight;  // likelihood
    };
    struct Neighborhood {
        int num_nb = 0;
        Neighbor* nb = NULL;
        ~Neighborhood() {
            if (nb) delete[] nb;
        }
    };

   protected:
    // help variables
    const int width;
    const int height;
    const int MAX_DISPARITY;
    cv::Mat D;

    cv::Mat spx;  // superpixel labels
    const int num_spx;
    uchar** adj_mat = NULL;

    // the superpixels after merged are denoted as regions
    cv::Mat rgn;  // region labels
    int num_rgn;
    int num_fp;  // number of front-parallel
    int num_occluded;
    int num_bm;  // number before merged

    SpxFeature* sfeature = NULL;
    RgnFeature* rfeature = NULL;

    cv::Vec3d* plane = NULL;

    cv::RNG rng;

   public:
    Parameters params;

    SurfaceFitting(cv::Mat D, SpxFeature* sfeature, cv::Mat spx,
                   const int num_spx, uchar** adj_mat, int maxDisparity,
                   Parameters params)
        : width(D.cols),
          height(D.rows),
          MAX_DISPARITY(maxDisparity),
          num_spx(num_spx),
          params(params) {
        this->D = D.clone();
        this->sfeature = sfeature;
        this->spx = spx.clone();
        this->adj_mat = adj_mat;

        rgn = cv::Mat(spx.size(), CV_32SC1);
        rfeature = new RgnFeature[num_spx];

        plane = new cv::Vec3d[num_spx];
    }

    ~SurfaceFitting() {
        if (rfeature) delete[] rfeature;
        if (plane) delete[] plane;
    }

    void process(cv::Mat& labeling, cv::Mat& refined_disp) {
        clock_t start, end;
        start = clock();

        refineAdjacentMat(sfeature, adj_mat);
        mergeFrontParallel(rgn, num_fp, num_rgn, adj_mat, sfeature, rfeature);

        int** group_id = new int*[num_rgn];
        for (int k = 0; k < num_rgn; k++)
            group_id[k] = new int[MAX_DISPARITY]{0};

        dispClassification(D, rgn, rfeature, group_id);

        planeFitting(rfeature, group_id, plane, D);
        end = clock();
        std::cout << "Time consumed in plane fitting: "
                  << (double)(end - start) / CLOCKS_PER_SEC << std::endl;

        for (int k = 0; k < num_rgn; k++) delete[] group_id[k];
        delete[] group_id;

        start = clock();

        planeFiltering(rfeature, plane, adj_mat, num_rgn, rgn, D);

        assignResults(plane, rgn, labeling, refined_disp);
        end = clock();
        std::cout << "Time consumed in plane filtering: "
                  << (double)(end - start) / CLOCKS_PER_SEC << std::endl;
    }

   protected:
    void refineAdjacentMat(SpxFeature* sfeature, uchar** adj_mat);

    void mergeFrontParallel(cv::Mat rgn, int& num_fp, int& num_rgn,
                            uchar** adj_mat, SpxFeature* sfeature,
                            RgnFeature* rfeature) {
        int* map = new int[num_spx]{0};

        computeFPMap(map, adj_mat, num_fp, num_bm, num_rgn);

        mapFrontParallel(map, rgn, adj_mat, sfeature, rfeature);

        delete[] map;
    }

    void computeFPMap(int* map, uchar** adj_mat, int& num_fp, int& num_bm,
                      int& num_rgn);
    void mapFrontParallel(int* map, cv::Mat rgn, uchar** adj_mat,
                          SpxFeature* sfeature, RgnFeature* rfeature) {
        mapAdjacentMat(map, adj_mat, num_bm);
        mapLabel(map, spx, rgn);
        mapFeature(map, sfeature, rfeature, num_bm, num_fp);
    }

    void mapAdjacentMat(int* map, uchar** adj_mat, const int& num_bm);
    void mapLabel(int* map, const cv::Mat& src, cv::Mat& dst);
    template <typename T>
    void mapFeature(int* map, T src, RgnFeature* dst, const int num_bm,
                    const int num_merge);

    void dispClassification(const cv::Mat& D, const cv::Mat& rgn,
                            RgnFeature* rfeature, int** group_id) {
        float** curve = new float*[num_rgn];

        for (int k = 0; k < num_rgn; k++)
            curve[k] = new float[MAX_DISPARITY]{0};
        computeCurve(D, rgn, curve);

        for (int k = 0; k < num_rgn; k++) {
            densityGroup(curve, rfeature, group_id, k);
        }

        for (int k = 0; k < num_rgn; k++) delete[] curve[k];
        delete[] curve;
    }

    void computeCurve(const cv::Mat& D, const cv::Mat& rgn, float** curve);
    void densityGroup(float** curve, RgnFeature* rfeature, int** group_id,
                      const int id);

    void planeFitting(RgnFeature* rfeature, int** group_id, cv::Vec3d* plane,
                      cv::Mat& D) {
        for (int k = 0; k < num_rgn; k++) {
            if (rfeature[k].occluded) {
                plane[k] = cv::Vec3d(0, 0, 0);
            } else {
                Group* gp = new Group[rfeature[k].group_num + 1];
                initGroup(rfeature[k], group_id[k], gp, D, k);
                groupSAC(rfeature[k], gp, plane[k], rng);
                delete[] gp;
            }
        }
    }

    void initGroup(RgnFeature& rfeature, int* group_id, Group* gp, cv::Mat& D,
                   const int& id);
    void disparityMeanVariance(RgnFeature& rfeature, Group* gp);
    void groupSAC(RgnFeature& rfeature, Group* gp, cv::Vec3d& plane,
                  cv::RNG& rng) {
        int num_cfg, ng = rfeature.group_num;

        if (ng == 1)
            num_cfg = 1;
        else if (ng == 2)
            num_cfg = 3;
        else
            num_cfg = cNM(ng, 1) + cNM(ng, 2) + cNM(ng, 3);
        Configuration* cfg = new Configuration[num_cfg];

        initConfiguration(gp, cfg, ng, num_cfg);

        disparityMeanVariance(rfeature, gp);

        double test_thrsh = 1;
        if (rfeature.ub - rfeature.lb >= 3 * params.bins_width) test_thrsh = 2;

        bool stop = false;
        int loop = 0;
        for (int k = 0; k < num_cfg; k++) {
            groupRANSAC(rfeature, gp, cfg[k], plane, rng, loop, stop, params,
                        test_thrsh);
            if (stop) break;
        }

        if (rfeature.inlier_rate > params.minimum_inlier_rate) {
            for (int thrsh = 4; thrsh >= 1; thrsh /= 2)
                rfeature.inlier_rate =
                    refineBestModel(rfeature, gp, plane, thrsh);
        } else {
            rfeature.occluded = true;
            plane = cv::Vec3d(0, 0, 0);
        }

        delete[] cfg;
    }

    double cNM(const int& n, const int& m);
    void initConfiguration(Group* gp, Configuration* cfg, const int ng,
                           int& num_cfg);
    void groupRANSAC(RgnFeature& rfeature, Group* gp, Configuration& cfg,
                     cv::Vec3d& plane, cv::RNG& rng, int& loop, bool& stop,
                     const Parameters& params, const double test_thrsh);
    double refineBestModel(RgnFeature& rfeature, Group* gp, cv::Vec3d& plane,
                           const double& thrsh);

    /**
     * The disparity plane refinement is denoted as planeFiltering here
     */
    void planeFiltering(RgnFeature* rfeature, cv::Vec3d* plane, uchar** adj_mat,
                        int& num_rgn, cv::Mat& rgn, const cv::Mat& D) {
        mergeOccludedRegion(rfeature, plane, adj_mat, num_rgn, rgn);

        // two iterations of disparity plane refinement
        for (int k = 0; k < 2; k++) {
            Neighborhood* nbh = new Neighborhood[num_rgn];

            computeNeighborWeight(nbh, rfeature, plane, adj_mat, num_rgn, rgn, D);
            planeWLS(nbh, rfeature, plane, num_rgn, rgn, params);
            // perform twice give the best result

            // planeMAP(nbh, rfeature, plane, num_rgn, params);
            // almost the same result for different iteration times

            // planeWMF(nbh, rfeature, plane, num_rgn, params);
            // perform once give the best result

            delete[] nbh;
        }
    }

    void mergeOccludedRegion(RgnFeature* rfeature, cv::Vec3d* plane,
                             uchar** adj_mat, int& num_rgn, cv::Mat& rgn) {
        int* map = new int[num_rgn]{0};

        computeORMap(map, rfeature, adj_mat, num_occluded, num_bm, num_rgn);
        mapOccludedRegion(map, rfeature, plane, adj_mat, rgn);

        delete[] map;
    }

    void computeORMap(int* map, RgnFeature* rfeature, uchar** adj_mat,
                      int& num_occluded, int& num_bm, int& num_rgn);
    void mapOccludedRegion(int* map, RgnFeature* rfeature, cv::Vec3d* plane,
                           uchar** adj_mat, cv::Mat& rgn) {
        mapAdjacentMat(map, adj_mat, num_bm);
        mapLabel(map, rgn, rgn);
        mapFeature(map, rfeature, rfeature, num_bm, num_occluded);
        mapPlane(map, plane, num_bm);
    }

    void mapPlane(int* map, cv::Vec3d* plane, int const& num_bm);

    void computeNeighborWeight(Neighborhood* nbh, RgnFeature* rfeature,
                               cv::Vec3d* plane, uchar** adj_mat,
                               const int& num_rgn, const cv::Mat& rgn,
                               const cv::Mat& D) {
        initNeighborhood(nbh, plane, adj_mat, num_rgn);
        computeTestWeight(nbh, rfeature, num_rgn, rgn, D);
        computePriorWeight(nbh, num_rgn);
    }

    void initNeighborhood(Neighborhood* nbh, cv::Vec3d* plane, uchar** adj_mat,
                          const int& num_rgn);
    void computeTestWeight(Neighborhood* nbh, RgnFeature* rfeature,
                           const int& num_rgn, const cv::Mat& rgn,
                           const cv::Mat& D);
    void computePriorWeight(Neighborhood* nbh, const int& num_rgn);

    void planeWLS(Neighborhood* nbh, RgnFeature* rfeature, cv::Vec3d* plane,
                  const int& num_rgn, const cv::Mat& rgn,
                  const Parameters& params);
    void planeWMF(Neighborhood* nbh, RgnFeature* rfeature, cv::Vec3d* plane,
                  const int& num_rgn, const Parameters& params);
    void planeMAP(Neighborhood* nbh, RgnFeature* rfeature, cv::Vec3d* plane,
                  const int& num_rgn, const Parameters& params);

    void assignResults(cv::Vec3d* plane, const cv::Mat& rgn, cv::Mat& labeling,
                       cv::Mat& refined_disp);
};

template <typename T>
void SurfaceFitting::mapFeature(int* map, T src, RgnFeature* dst,
                                const int num_bm, const int num_merge) {
    RgnFeature* tfeature = new RgnFeature[num_rgn];
    for (int k = 0; k < num_merge; k++) {
        tfeature[k].tl = cv::Point2i(height, width);
        tfeature[k].br = cv::Point2i(0, 0);
    }
    for (int k = 0; k < num_bm; k++) {
        int id = map[k];
        tfeature[id].color = (tfeature[id].color * tfeature[id].num_pixel +
                              src[k].color * src[k].num_pixel) /
                             (tfeature[id].num_pixel + src[k].num_pixel);

        if (id < num_merge) {
            tfeature[id].br.x = src[k].br.x > tfeature[id].br.x
                                    ? src[k].br.x : tfeature[id].br.x;
            tfeature[id].br.y = src[k].br.y > tfeature[id].br.y
                                    ? src[k].br.y : tfeature[id].br.y;
            tfeature[id].tl.x = src[k].tl.x < tfeature[id].tl.x
                                    ? src[k].tl.x : tfeature[id].tl.x;
            tfeature[id].tl.y = src[k].tl.y < tfeature[id].tl.y
                                    ? src[k].tl.y : tfeature[id].tl.y;
        } else {
            tfeature[id].br = src[k].br;
            tfeature[id].tl = src[k].tl;
        }
        tfeature[id].num_pixel += src[k].num_pixel;
        tfeature[id].disp = src[k].disp;
        tfeature[id].mean = src[k].mean;
        tfeature[id].occluded = src[k].occluded;
    }
    for (int k = 0; k < num_rgn; k++) dst[k] = tfeature[k];
    delete[] tfeature;
}
