#include "SurfaceFitting.h"
#include <Eigen/Dense>

/**
 * Define the 3D neighborhood here
 */
void SurfaceFitting::refineAdjacentMat(SpxFeature * sfeature,  uchar ** adj_mat)
{
    for (int i = 0; i < num_spx; i++)
        for (int j = 0; j < num_spx; j++)
            if (adj_mat[i][j]) {
                if (abs(sfeature[i].mean - sfeature[j].mean) == 0) adj_mat[i][j] = 1;
                else if (abs(sfeature[i].mean - sfeature[j].mean) == 1) adj_mat[i][j] = 2;
                else
                    adj_mat[i][j] = 3;  // 3: 3D-neighbor, 2: 2D-neighbor
            }
}

/**
 * help functions
 */
bool is_front_parallel(const int& id, uchar** adj_mat, const int& num_spx) {
    bool flag = true;
    for (int i = 0; i < num_spx; i++)
        if (adj_mat[id][i] > 1) flag = false;
    return flag;
}
void recurse_front_parallel(uchar* is_seed, uchar* is_traversed,
                            int const& seed_id, int const& plane_id,
                            uchar** adj_mat, int* map, const int& num_spx) {
    is_traversed[seed_id] = 1;
    for (int i = 0; i < num_spx; i++)
        if (adj_mat[seed_id][i] && !is_traversed[i]) {
            map[i] = plane_id;
            if (is_seed[i] && !is_traversed[i]) {
                recurse_front_parallel(is_seed, is_traversed, i, plane_id,
                                       adj_mat, map, num_spx);
            }
            is_traversed[i] = 1;
        }
}
/**
 * Compute front-parall map
 * to merge superpixels with same mean disparities to a new superpixel
 * The superpixels after merged are denoted as regions
 */
void SurfaceFitting::computeFPMap(int* map, uchar** adj_mat, int& num_fp,
                                  int& num_bm, int& num_rgn) {
    uchar* is_traversed = new uchar[num_spx]{ 0 };
    uchar* is_seed = new uchar[num_spx]{ 0 };
    for (int i = 0; i < num_spx; i++)
        if (is_front_parallel(i, adj_mat, num_spx)) is_seed[i] = 1;
    int num_region = 0;
    for (int i = 0; i < num_spx; i++)
        if (is_seed[i] && !is_traversed[i]) {
            map[i] = num_region;
            recurse_front_parallel(is_seed, is_traversed, i, num_region,
                                   adj_mat, map, num_spx);
            num_region += 1;
        }
    num_fp = num_region;
    for (int i = 0; i < num_spx; i++)
        if (!is_traversed[i]) {
            map[i] = num_region;
            num_region += 1;
        }
    num_bm = num_spx;
    num_rgn = num_region;

    delete[] is_seed;
    delete[] is_traversed;
}

/**
 * After merging, map the properties of superpixels to new indices
 */
void SurfaceFitting::mapAdjacentMat(int* map, uchar** adj_mat,
                                    const int& num_bm) {
    uchar** adj_mat_new = new uchar*[num_spx];
    for (int i = 0; i < num_spx; i++)
        adj_mat_new[i] = new uchar[num_spx]{ 0 };
    for (int i = 0; i < num_bm; i++)
        for (int j = 0; j < num_bm; j++)
            if (adj_mat[i][j])
                adj_mat_new[map[i]][map[j]] = adj_mat[i][j];
    for (int i = 0; i < num_spx; i++)
        for (int j = 0; j < num_spx; j++)
            adj_mat[i][j] = adj_mat_new[i][j];

    for (int i = 0; i < num_spx; i++)
        delete[] adj_mat_new[i];
    delete[] adj_mat_new;
}
void SurfaceFitting::mapLabel(int* map, const cv::Mat& src, cv::Mat& dst) {
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++) {
            int tl = src.at<int>(i, j);
            dst.at<int>(i, j) = map[tl];
        }
}

/**
 * Compute the disparity distributions
 */
void SurfaceFitting::computeCurve(const cv::Mat& D, const cv::Mat& rgn,
                                  float** curve) {
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++){
            int tl = rgn.at<int>(i, j);
            int disp = D.at<float>(i, j);
            curve[tl][disp] += 1;
        }
}

/**
 * Compute the disparity density distributions
 */
void computeDensity(float* curve, float* density, const int MAX_DISPARITY,
                    const int bins_width, const int dc = 1) {
    for (int k = -dc; k <= dc; k++) {
        if (k + bins_width < 1) continue;
        density[bins_width] += curve[k + bins_width];
    }
    for (int k = bins_width + 1; k < MAX_DISPARITY; k++) {
        if (k - 1 - dc < 1) density[k] = density[k - 1] + curve[k + dc];
        else if (k + dc >= MAX_DISPARITY) density[k] = density[k - 1] - curve[k - 1 - dc];
        else density[k] = density[k - 1] + curve[k + dc] - curve[k - 1 - dc];
    }
}
/**
 * Modify the density distribution to make it have a unique maximum
 */
void modifyDensity(float* density, const int MAX_DISPARITY,
                   const int bins_width) {
    for (int k = bins_width + 1; k < MAX_DISPARITY; k++) {
        int rn = 0;
        if (density[k + 1] == density[k]) {
            for (int t = k; t < MAX_DISPARITY - 1; t++) {
                if (density[t + 1] == density[t]) rn += 1;
                else break;
            }
            density[k + rn / 2] += FLT_MIN;
        }
        k += rn;
    }
}
/**
 * This corresponds to reliable observation selection
 * Compute the disparity range of raliable observations
 */
bool modifyDisparityBound(RgnFeature& rfeature, float* density,
                          const int MAX_DISPARITY, const int bins_width) {
    int lb = bins_width, ub = MAX_DISPARITY - 1;

    double densi_mean(0);
    for (int i = lb; i <= ub; i++)
        densi_mean += density[i];
    densi_mean /= (ub - lb + FLT_MIN);
    if (densi_mean <= FLT_EPSILON || rfeature.mean == 0) {
        rfeature.occluded = true;
        return false;
    }

    int disp_center = rfeature.disp;
    if (density[disp_center] < densi_mean) {
        rfeature.occluded = true;
        return false;
    }
    if (!rfeature.occluded) {
        for (int i = disp_center; i <= ub; i++)
            if (density[i] < densi_mean) {
                ub = i - 1;
                break;
            }
        for (int i = disp_center; i >= lb; i--)
            if (density[i] < densi_mean) {
                lb = i + 1;
                break;
            }
        rfeature.ub = ub;
        rfeature.lb = lb;
        return true;
    }
    else return false;
}

/**
 * help functions for the density based disparity group
 */
void maxDensityDelta(RgnFeature& rfeature, float* density, float* delta,
                     double& densi_max, double& delta_max, int& max_id) {
    for (int i = rfeature.lb; i <= rfeature.ub; i++)
        if (density[i] > densi_max) {
            densi_max = density[i];
            max_id = i;
        }
    delta_max = std::max(rfeature.ub - max_id, max_id - rfeature.lb);
    delta[max_id] = delta_max;
}
void computeDelta(RgnFeature& rfeature, float* density, float* delta,
                  const int max_id) {
    for (int i = rfeature.lb; i <= rfeature.ub; i++) {
        if (i == max_id) continue;
        for (int k = 1; k < rfeature.ub - rfeature.lb + 1; k++) {
            if (i - k >= rfeature.lb && density[i - k] > density[i]) {
                delta[i] = k;
                break;
            }
            if (i + k <= rfeature.ub && density[i + k] > density[i]) {
                delta[i] = k;
                break;
            }
        }
    }
}
void findGroupCenter(RgnFeature& rfeature, float* density, float* delta,
                     int* is_center, const double densi_minimum,
                     const double delta_minimum, const int max_id) {
    int cn = 0;
    for (int k = rfeature.lb; k <= rfeature.ub; k++)
        if ((density[k] > densi_minimum &&
             delta[k] + DBL_MIN > delta_minimum) ||
            (k == max_id)) {
            cn += 1;
            is_center[k] = cn;
        }
    rfeature.group_num = cn;
}
void assignGroupMember(RgnFeature& rfeature, int* group_id, int* is_center,
                       float* density) {
    int lb = rfeature.lb, ub = rfeature.ub;
    if (rfeature.group_num == 1) {
        for (int k = lb; k <= ub; k++)
            group_id[k] = 1;
    }
    else {
        for (int k = lb; k <= ub; k++) {
            if (is_center[k]) {
                group_id[k] = is_center[k];
                continue;
            }
            for (int t = 1; t < ub - lb + 1; t++) {
                if (k - t >= lb && is_center[k - t] && density[k - t] > density[k]) {
                    group_id[k] = is_center[k - t];
                    break;
                }
                if (k + t <= ub && is_center[k + t] && density[k + t] > density[k]) {
                    group_id[k] = is_center[k + t];
                    break;
                }
            }
        }
    }
}
/**
 * Density based disparity group
 * prepare groups for the GroupSAC
 */
void SurfaceFitting::densityGroup(float** curve, RgnFeature* rfeature,
                                  int** group_id, const int id) {
    float* density = new float[MAX_DISPARITY] {0};
    float* delta = new float[MAX_DISPARITY] {0};

    computeDensity(curve[id], density, MAX_DISPARITY, params.bins_width,
                   params.bins_width);
    modifyDensity(density, MAX_DISPARITY, params.bins_width);
    if (!modifyDisparityBound(rfeature[id], density, MAX_DISPARITY, params.bins_width)) return;
    double densi_max(0), delta_max(0);
    int max_id(0);
    maxDensityDelta(rfeature[id], density, delta, densi_max, delta_max, max_id);

    computeDelta(rfeature[id], density, delta, max_id);

    int* is_center = new int[MAX_DISPARITY]{ 0 };
    findGroupCenter(rfeature[id], density, delta, is_center, densi_max / 3.0,
                    3 * params.bins_width - 1, max_id);

    assignGroupMember(rfeature[id], group_id[id], is_center, density);

    delete[] density;
    delete[] delta;
    delete[] is_center;
}

/**
 * Initialize groups
 * former operation just compute the disparity range of each group
 * This step assigns all observation points to each group
 */
void SurfaceFitting::initGroup(RgnFeature& rfeature, int* group_id, Group* gp,
                               cv::Mat& D, const int& id) {
    for (int k = 0; k < MAX_DISPARITY; k++)
        gp[group_id[k]].disp_num += 1;
    for (int i = rfeature.tl.x; i <= rfeature.br.x; i++)
        for (int j = rfeature.tl.y; j <= rfeature.br.y; j++)
            if (rgn.at<int>(i, j) == id) {
                gp[group_id[(int)D.at<float>(i, j)]].points_num += 1;
            }
    rfeature.num_pixel = 0;
    for (int k = 1; k <= rfeature.group_num; k++) {
        rfeature.num_pixel += gp[k].points_num;
        cv::Point3i* t_points = new cv::Point3i[gp[k].points_num];
        gp[k].points = t_points;
        gp[k].points_num = 0;
    }
    rfeature.num_pixel += gp[0].points_num;
    int gp_id, tn, d;
    for (int i = rfeature.tl.x; i <= rfeature.br.x; i++)
        for (int j = rfeature.tl.y; j <= rfeature.br.y; j++)
            if (rgn.at<int>(i, j) == id) {
                d = D.at<float>(i, j);
                gp_id = group_id[d];
                if (gp_id) {
                    tn = gp[gp_id].points_num;
                    gp[gp_id].points[tn] = cv::Point3i(i, j, d);
                    gp[gp_id].points_num += 1;
                }
            }
}

/**
 * MAP inference of disparity mean and variance
 */
void SurfaceFitting::disparityMeanVariance(RgnFeature& rfeature, Group* gp) {
    int num(0);
    int ng = rfeature.group_num;
    double mean(0), variance(0);

    for (int i = 1; i < ng + 1; i++) {
        for (int j = 0; j < gp[i].points_num; j++) {
            cv::Point3i p = gp[i].points[j];
            mean += p.z;
            variance += p.z*p.z;
            num += 1;
        }
    }
    mean = mean / num;
    variance = variance / num - mean*mean;

    mean = (mean*num + rfeature.disp * num) / (num + num);
    variance = (variance * num + 2 +
                (mean - rfeature.disp) * (mean - rfeature.disp) * num) /
               (num + 3 + 2);

    rfeature.disp_mean = mean;
    rfeature.disp_var = variance;
}

/**
 * Compute combinations C_n^m
 */
double SurfaceFitting::cNM(const int& n, const int& m) {
    double c = 1;
    if (m > n) return 0;
    if (m == 0) return 1;
    for (int i = 0; i < m; i++)
        c *= (n - i);
    for (int i = m; i > 0; i--)
        c /= i;
    return c;
}

/**
 * help function for GroupSAC
 * initialize configurations
 */
void SurfaceFitting::initConfiguration(Group* gp, Configuration* cfg,
                                       const int ng, int& num_cfg) {
    num_cfg = 0;
    if (ng >= 1) {
        for (int i = 1; i < ng + 1; i++) {
            cfg[num_cfg].group_num = 1;
            cfg[num_cfg].group_id[0] = i;
            cfg[num_cfg].disp_num = gp[i].disp_num;
            cfg[num_cfg].points_num = gp[i].points_num;
            cfg[num_cfg].sample_num = cNM(gp[i].points_num, 3);
            num_cfg += 1;
        }
    }
    if (ng >= 2) {
        for (int i = 1; i < ng; i++)
            for (int j = i + 1; j < ng + 1; j++) {
                cfg[num_cfg].group_num = 2;
                cfg[num_cfg].group_id[0] = i;
                cfg[num_cfg].group_id[1] = j;
                cfg[num_cfg].disp_num = gp[i].disp_num + gp[j].disp_num;
                cfg[num_cfg].points_num = gp[i].points_num + gp[j].points_num;
                cfg[num_cfg].sample_num =
                    cNM(gp[i].points_num, 1) * cNM(gp[j].points_num, 2) +
                    cNM(gp[i].points_num, 2) * cNM(gp[j].points_num, 1);
                num_cfg += 1;
            }
    }
    if (ng >= 3) {
        for (int i = 1; i < ng - 1; i++)
            for (int j = i + 1; j < ng; j++)
                for (int k = j + 1; k < ng + 1; k++) {
                    cfg[num_cfg].group_num = 3;
                    cfg[num_cfg].group_id[0] = i;
                    cfg[num_cfg].group_id[1] = j;
                    cfg[num_cfg].group_id[2] = k;
                    cfg[num_cfg].disp_num =
                        gp[i].disp_num + gp[j].disp_num + gp[k].disp_num;
                    cfg[num_cfg].points_num =
                        gp[i].points_num + gp[j].points_num + gp[k].points_num;
                    cfg[num_cfg].sample_num = cNM(gp[i].points_num, 1) *
                                              cNM(gp[j].points_num, 1) *
                                              cNM(gp[k].points_num, 1);
                    num_cfg += 1;
                }
    }
    std::sort(cfg, cfg + num_cfg);
}

/**
 * help functions for GroupSAC
 * sample three points to generate a plane model for different type of
 * configurations
 */
void sampleMinimumPoints(SurfaceFitting::Group& g0, cv::Point3i* p,
                         cv::RNG& rng) {
    int tn = g0.points_num;
    // generate three different numbers from [0,tn)
    int p0, p1, p2;
    p0 = rng.operator() (tn);
    p1 = rng.operator() (tn - 1);
    p2 = rng.operator() (tn - 2);
    if (p1 == p0) p1 = tn - 1;
    if (p2 == p0) {
        if (p1 == tn - 2) p2 = tn - 1;
        else p2 = tn - 2;
    }
    else if (p2 == p1) {
        if (p0 == tn - 2) p2 = tn - 1;
        else p2 = tn - 2;
    }
    p[0] = g0.points[p0];
    p[1] = g0.points[p1];
    p[2] = g0.points[p2];
}
void sampleMinimumPoints(SurfaceFitting::Group& g0, SurfaceFitting::Group& g1,
                         cv::Point3i* p, cv::RNG& rng) {
    int t0 = g0.points_num, t1 = g1.points_num;
    int p0, p1, p2;
    p0 = rng.operator() (t0 + t1);
    if (p0 < t0) {
        p[0] = g0.points[p0];
        p1 = rng.operator() (t0 - 1);
        if (p1 == p0) p1 = t0 - 1;
        p[1] = g0.points[p1];
        p2 = rng.operator() (t1);
        p[2] = g1.points[p2];
    }
    else {
        p[0] = g1.points[p0 - t0];
        p1 = rng.operator() (t1 - 1);
        if (p1 == p0 - t0) p1 = t1 - 1;
        p[1] = g1.points[p1];
        p2 = rng.operator() (t0);
        p[2] = g0.points[p2];
    }
}
void sampleMinimumPoints(SurfaceFitting::Group& g0, SurfaceFitting::Group& g1,
                         SurfaceFitting::Group& g2, cv::Point3i* p,
                         cv::RNG& rng) {
    int t0 = g0.points_num, t1 = g1.points_num, t2 = g2.points_num;
    int p0, p1, p2;
    p0 = rng.operator() (t0);
    p1 = rng.operator() (t1);
    p2 = rng.operator() (t2);
    p[0] = g0.points[p0];
    p[1] = g1.points[p1];
    p[2] = g2.points[p2];
}
/**
 * Generate plane model using least-squares (LS)
 */
bool generatePlaneModel(cv::Point3i* p, cv::Vec3d& plane, const int& num) {
    Eigen::MatrixXf A(num, 3);
    Eigen::VectorXf B(num, 1);
    Eigen::VectorXf X(3, 1);
    for (int k = 0; k < num; k++) {
        A(k, 0) = p[k].x;
        A(k, 1) = p[k].y;
        A(k, 2) = 1.f;
        B(k) = p[k].z;
    }
    X = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);
    plane[0] = X(0);
    plane[1] = X(1);
    plane[2] = X(2);
    return true;
}
/**
 * Test the plane model and compute inlier rate
 */
double testPlaneModel(SurfaceFitting::Group* gp, const cv::Vec3d& plane,
                      const int& ng, const double& thrsh) {
    //test on all groups
    int inlier_num(0), total_num(0);
    for (int i = 1; i < ng + 1; i++) {
        for (int j = 0; j < gp[i].points_num; j++) {
            cv::Point3i p = gp[i].points[j];
            double disp = plane[0]*p.x + plane[1]*p.y + plane[2];
            if (abs(disp - p.z) <= thrsh)
                inlier_num ++;
            total_num++;
        }
    }
    // remove the points which are used to compute the model
    inlier_num -= 3;
    return inlier_num / (double)total_num;
}
/**
 * Degeneracy test
 * by comparing mean and variance
 */
bool testDegeneracy(RgnFeature& rfeature, SurfaceFitting::Group* gp,
                    const cv::Vec3d& plane, const Parameters& params) {
    int num(0);
    int ng = rfeature.group_num;
    double mean(0), variance(0);

    for (int i = 1; i < ng + 1; i++) {
        for (int j = 0; j < gp[i].points_num; j++) {
            cv::Point3i p = gp[i].points[j];
            double disp = plane[0]*p.x + plane[1]*p.y + plane[2];
            mean += disp;
            variance += disp*disp;
            num += 1;
        }
    }
    mean = mean / num;
    variance = variance / num - mean*mean;

    if (abs(mean - rfeature.disp_mean) > params.bins_width ||
        abs(variance - rfeature.disp_var) >
            params.bins_width * params.bins_width)
        return false;
    else
        return true;
}
/**
 * The overall GroupSAC procedure
 */
void SurfaceFitting::groupRANSAC(RgnFeature& rfeature, Group* gp,
                                 Configuration& cfg, cv::Vec3d& plane,
                                 cv::RNG& rng, int& loop, bool& stop,
                                 const Parameters& params,
                                 const double test_thrsh) {
    const double eta_zero = params.eta_zero;

    cv::Vec3d best_pl;
    cv::Point3i p[3];

    const int ng = cfg.group_num;

    double inlier_rate(0), tmp_rate(0);

    int max_loop = 50 * cfg.sample_num / cNM(cfg.points_num, 3) + loop;
    int max_trials = INT_MAX;

    for (; loop < max_loop; loop++) {
        // sample
        if (ng == 1) sampleMinimumPoints(gp[cfg.group_id[0]], p, rng);
        else if (ng == 2)
            sampleMinimumPoints(gp[cfg.group_id[0]], gp[cfg.group_id[1]], p,
                                rng);
        else if (ng == 3)
            sampleMinimumPoints(gp[cfg.group_id[0]], gp[cfg.group_id[1]],
                                gp[cfg.group_id[2]], p, rng);
        // generate model
        if (!generatePlaneModel(p, plane, 3)) {
            std::cout << "can not generate model!" << std::endl;
            continue;
        }
        // test the model
        tmp_rate = testPlaneModel(gp, plane, rfeature.group_num, test_thrsh);
        if (tmp_rate > inlier_rate) {
            inlier_rate = tmp_rate;
            best_pl = plane;
            // update max_trials
            max_trials = log(eta_zero) / log(1 - pow(inlier_rate, 3));
        }
        if (loop >= max_trials) {
            if (testDegeneracy(rfeature, gp, plane, params)) {
                stop = true;
                break;
            }
        }
    }
    rfeature.inlier_rate = inlier_rate;
    plane = best_pl;
}

/**
 * Refine the RANSAC fitting results
 * by fiiting a new model using the inlier points
 */
double SurfaceFitting::refineBestModel(RgnFeature& rfeature, Group* gp,
                                       cv::Vec3d& plane, const double& thrsh) {
    cv::Point3i* points=new cv::Point3i[rfeature.num_pixel];
    //test on all groups
    int inlier_num(0), total_num(0);
    for (int i = 1; i < rfeature.group_num + 1; i++) {
        for (int j = 0; j < gp[i].points_num; j++) {
            cv::Point3i p = gp[i].points[j];
            double disp = plane[0] * p.x + plane[1] * p.y + plane[2];
            if (abs(disp - p.z) <= thrsh) {
                points[inlier_num] = p;
                inlier_num++;
            }
            total_num++;
        }
    }

    if (inlier_num >= 3) generatePlaneModel(points, plane, inlier_num);
    // remove the points which are used to compute the model
    inlier_num -= 3;
    return inlier_num / (double)total_num;

    delete[] points;
}

/**
 * help functions for merging occluded regions
 */
bool isOccludedNeighbor(RgnFeature* rfeature, uchar** adj_mat, const int& id,
                        const int& i) {
    return (adj_mat[id][i] && adj_mat[id][i] <= 2 && rfeature[i].occluded);
}
bool isOccluded(RgnFeature* rfeature, uchar** adj_mat, const int& num_rgn,
                const int& id) {
    bool flag = false;
    if (rfeature[id].occluded)
        for (int i = 0; i < num_rgn; i++)
            if (isOccludedNeighbor(rfeature, adj_mat, id, i)) flag = true;
    return flag;
}
void recurseOccluded(RgnFeature* rfeature, int* map, uchar** adj_mat,
                     const int& num_rgn, uchar* is_seed, uchar* is_traversed,
                     const int& seed_id, const int& region_id) {
    is_traversed[seed_id] = 1;
    for (int i = 0; i < num_rgn; i++)
        if (isOccludedNeighbor(rfeature, adj_mat, seed_id, i) &&
            !is_traversed[i]) {
            map[i] = region_id;
            recurseOccluded(rfeature, map, adj_mat, num_rgn, is_seed,
                            is_traversed, i, region_id);
            is_traversed[i] = 1;
        }
}
/**
 * Merge the occluded regions and compute their new indices
 */
void SurfaceFitting::computeORMap(int* map, RgnFeature* rfeature,
                                  uchar** adj_mat, int& num_occluded,
                                  int& num_bm, int& num_rgn) {
    uchar* is_seed = new uchar[num_rgn]{ 0 };
    uchar* is_traversed = new uchar[num_rgn]{ 0 };

    for (int i = 0; i < num_rgn; i++)
        if (isOccluded(rfeature, adj_mat, num_rgn, i)) is_seed[i] = 1;
    int num_region = 0;
    for (int i = 0; i < num_rgn; i++)
        if (is_seed[i] && !is_traversed[i]) {
            map[i] = num_region;
            recurseOccluded(rfeature, map, adj_mat, num_rgn, is_seed,
                            is_traversed, i, num_region);
            num_region += 1;
        }
    num_occluded = num_region;

    for (int i = 0; i < num_rgn; i++)
        if (!is_traversed[i]) {
            map[i] = num_region;
            num_region += 1;
        }
    num_bm = num_rgn;
    num_rgn = num_region;

    delete[] is_seed;
    delete[] is_traversed;
}

/**
 * Map the plane label to new indices
 */
void SurfaceFitting::mapPlane(int* map, cv::Vec3d* plane, const int& num_bm) {
    cv::Vec3d* tp = new cv::Vec3d[num_spx];

    for (int i = 0; i < num_bm; i++) {
        int id = map[i];
        tp[id] = plane[i];
    }

    for (int i = 0; i < num_spx; i++)
        plane[i] = tp[i];

    delete[] tp;
}

/**
 * Initialize the 3D neighborhood
 */
void SurfaceFitting::initNeighborhood(Neighborhood* nbh, cv::Vec3d* plane,
                                      uchar** adj_mat, const int& num_rgn) {
    for (int i = 0; i < num_rgn; i++) {
        int* nb_list = new int[num_rgn] {0};

        int num_nb = 1;// the region itself is in the neighborhood
        nb_list[0] = i;

        for (int j = 0; j < num_rgn; j++) {
            if (j == i) continue;
            // 3D neighbors
            if (adj_mat[i][j] && adj_mat[i][j] <= 2) {
                nb_list[num_nb] = j;
                num_nb += 1;
            }
        }

        // in case of bugs
        if (num_nb == 1) {
            // if no 3D-neighbor besides itself
            // then add all 2D-neighbor to its neighborhood
            // because a region must have 2D neighbors in image space.
            // in such case, we may add some occluded region as neighbors.
            for (int j = 0; j < num_rgn; j++) {
                if (j == i) continue;
                if (adj_mat[i][j]) {
                    nb_list[num_nb] = j;
                    num_nb += 1;
                }
            }
        }

        nbh[i].num_nb = num_nb;
        nbh[i].nb = new Neighbor[num_nb];

        for (int k = 0; k < num_nb; k++) {
            nbh[i].nb[k].id = nb_list[k];
            nbh[i].nb[k].pl = plane[nb_list[k]];
        }

        delete[] nb_list;
    }
}

/**
 * Test the plane model
 * and compute the inlier ratio as likelihood
 */
double testPlaneModel(const cv::Vec3d& plane, cv::Point3i* points,
                      int const& tn, double const& thrsh) {
    int inlier_num(0);
    for (int i = 0; i < tn; i++) {
        cv::Point3i p = points[i];
        if (abs(plane[0] * p.x + plane[1] * p.y + plane[2] - p.z) <= thrsh)
            inlier_num++;
    }
    return inlier_num / (double)tn;
}
/**
 * Compute the likelihood of N_3d(s) tested on superpixel s
 */
void SurfaceFitting::computeTestWeight(Neighborhood* nbh, RgnFeature* rfeature,
                                       const int& num_rgn, const cv::Mat& rgn,
                                       const cv::Mat& D) {
    for (int id = 0; id < num_rgn; id++) {
        int tn = 0;
        cv::Point3i* points = new cv::Point3i[rfeature[id].num_pixel];

        // get all the points in the region
        for (int i = rfeature[id].tl.x; i <= rfeature[id].br.x; i++)
            for (int j = rfeature[id].tl.y; j <= rfeature[id].br.y; j++)
                if (rgn.at<int>(i, j) == id) {
                    int d = D.at<float>(i, j);
                    points[tn] = cv::Point3i(i, j, d);
                    tn += 1;
                }

        double thrsh = 1;

        // compute test weight for all neighbors
        for (int k = 0; k < nbh[id].num_nb; k++)
            nbh[id].nb[k].test_weight = testPlaneModel(nbh[id].nb[k].pl, points, tn, thrsh);
        if (rfeature[id].occluded) nbh[id].nb[0].test_weight = 0;

        delete[] points;
    }
}

/**
 * The prior weight is superpixel s tested on itself
 */
void SurfaceFitting::computePriorWeight(Neighborhood* nbh, const int& num_rgn) {
    for (int i = 0; i < num_rgn; i++) {
        for (int k = 0; k < nbh[i].num_nb; k++) {
            int id = nbh[i].nb[k].id;
            nbh[i].nb[k].prior_weight = nbh[id].nb[0].test_weight;
        }
    }
}

/**
 * Obtain pixels located on boundary of superpixels
 */
void computeEdgeMap(cv::Mat& edge_rgn, const cv::Mat& rgn, int* num_ep,
                    int** edge_length) {
    const int dx8[8] = { -1, -1,-1, 0, 0, 1, 1, 1 };
    const int dy8[8] = { -1, 0, 1,-1, 1,-1, 0, 1 };

    int width = rgn.cols;
    int height = rgn.rows;

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++) {
            int cl = rgn.at<int>(i, j);
            for (int k = 0; k < 8; k++) {
                int x = i + dx8[k];
                int y = j + dy8[k];
                if ((x >= 0 && x < height) && (y >= 0 && y < width)) {
                    int tl = rgn.at<int>(x, y);
                    if (tl != cl) {
                        edge_rgn.at<int>(i, j) = cl;
                        num_ep[cl] += 1;
                        edge_length[tl][cl] += 1;
                        edge_length[cl][tl] += 1;
                        break;
                    }
                }
            }
        }
}
/**
 * Compute color similarity weight
 */
void computeColorWeight(SurfaceFitting::Neighborhood* nbh, RgnFeature* rfeature,
                        double* weight, const int& i,
                        const Parameters& params) {
    int num_nb = nbh[i].num_nb;
    int id_0 = nbh[i].nb[0].id;

    for (int k = 0; k < num_nb; k++) {
        int id = nbh[i].nb[k].id;

        cv::Vec3f colordiff = rfeature[id].color - rfeature[id_0].color;
        double omega_ij = sqrt(colordiff.ddot(colordiff));
        omega_ij = params.lambda * std::max(exp(-omega_ij / params.gamma),
                                            0.01 /*(double)FLT_MIN*/);

        weight[k] = omega_ij;
    }
}
/**
 * Compute posteriori weights of N_3d(s) to superpixel s
 */
void computePosterioriWeight(SurfaceFitting::Neighborhood* nbh,
                             RgnFeature* rfeature, double* weight, const int& i,
                             const Parameters& params) {
    int id, num_nb = nbh[i].num_nb;

    double total_weight(0);

    double* color_weight = new double[num_nb];
    computeColorWeight(nbh, rfeature, color_weight, i, params);

    for (int k = 0; k < nbh[i].num_nb; k++) {
        id = nbh[i].nb[k].id;
        total_weight += color_weight[k] *
                        std::max(nbh[i].nb[k].test_weight, 0.01) *
                        nbh[i].nb[k].prior_weight;
    }
    if (total_weight > FLT_EPSILON) {
        for (int k = 0; k < nbh[i].num_nb; k++) {
            id = nbh[i].nb[k].id;
            weight[k] = color_weight[k] *
                        std::max(nbh[i].nb[k].test_weight, 0.01) *
                        nbh[i].nb[k].prior_weight / total_weight;
        }
    }
    // in case of bugs
    else {
        for (int k = 0; k < nbh[i].num_nb; k++)
            weight[k] = 1.0 / nbh[i].num_nb;
    }
}
/**
 * Posterior predictive probability as the weight
 */
double pdWeight(SurfaceFitting::Neighborhood& nbh, double* posteriori_weight,
                cv::Point3d& p) {
    double weight(0);
    double thrsh = 1;
    int num_nb = nbh.num_nb;
    for (int i = 0; i < num_nb; i++) {
        cv::Vec3d plane = nbh.nb[i].pl;
        if (abs(plane[0]*p.x + plane[1]*p.y + plane[2] - p.z) <= thrsh)
            weight += posteriori_weight[i];
    }
    return weight;
}
/**
 * Generate the plane model using weighted-least-squares (WLS)
 */
void generatePlaneModel(cv::Point3d* p, cv::Vec3d& plane, int const& num,
                        double* weight) {
    Eigen::MatrixXf A(num, 3);
    Eigen::VectorXf B(num, 1);
    Eigen::VectorXf X(3, 1);
    for (int k = 0; k < num; k++) {
        A(k, 0) = weight[k] * p[k].x;
        A(k, 1) = weight[k] * p[k].y;
        A(k, 2) = weight[k] * 1.f;
        B(k) = weight[k] * p[k].z;
    }
    X = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);
    plane[0] = X(0);
    plane[1] = X(1);
    plane[2] = X(2);
}
/**
 * The overall disparity plane refinement procedure
 */
void SurfaceFitting::planeWLS(Neighborhood* nbh, RgnFeature* rfeature,
                              cv::Vec3d* plane, const int& num_rgn,
                              const cv::Mat& rgn, const Parameters& params) {
    int* num_ep = new int[num_rgn]{0};  // number of edge (boundary) pixels

    int** edge_length = new int*[num_rgn];
    for (int k = 0; k < num_rgn; k++)
        edge_length[k] = new int[num_rgn] {0};

    cv::Mat edge_rgn(height, width, CV_32SC1, cv::Scalar::all(-1));
    computeEdgeMap(edge_rgn, rgn, num_ep, edge_length);

    for (int k = 0; k < num_rgn; k++) delete[] edge_length[k];
    delete[] edge_length;

    for (int i = 0; i < num_rgn; i++) {
        if (rfeature[i].pd_weight > 0.9)
            continue;  // This to save computational time in further
                       // iterations

        const int num_nb = nbh[i].num_nb;

        double* posteriori_weight = new double[num_nb];
        computePosterioriWeight(nbh, rfeature, posteriori_weight, i, params);

        int num_p = 0;
        for (int k = 0; k < num_nb; k++) {
            num_p += num_ep[nbh[i].nb[k].id];
        }

        double* weight = new double[num_p];
        cv::Point3d* p = new cv::Point3d[num_p];

        num_p = 0;
        for (int k = 0; k < num_nb; k++) {
            int id = nbh[i].nb[k].id;
            cv::Vec3d tmp_pl = nbh[i].nb[k].pl;

            for (int x = rfeature[id].tl.x; x <= rfeature[id].br.x; x++)
                for (int y = rfeature[id].tl.y; y <= rfeature[id].br.y; y++)
                    if (edge_rgn.at<int>(x, y) == id) {
                        p[num_p].x = x;
                        p[num_p].y = y;
                        p[num_p].z = x*tmp_pl[0] + y*tmp_pl[1] + tmp_pl[2];

                        weight[num_p] =
                            pdWeight(nbh[i], posteriori_weight, p[num_p]);

                        num_p += 1;
                    }
        }

        double pd_weight_mean(0);
        for (int k = 0; k < num_p; k++)
            pd_weight_mean += weight[k] / num_p;

        // if the pd_weight is larger than the former iteration
        // this is to save the computational time
        if (num_p >= 3 && pd_weight_mean > rfeature[i].pd_weight) {
            rfeature[i].pd_weight = pd_weight_mean;
            generatePlaneModel(p, plane[i], num_p, weight);
        }

        delete[] p;
        delete[] weight;
        delete[] posteriori_weight;
    }

    delete[] num_ep;
}

/**
 * Disparity plane refinement using weighted-mean-filtering (WMF)
 */
void SurfaceFitting::planeWMF(Neighborhood* nbh, RgnFeature* rfeature,
                              cv::Vec3d* plane, const int& num_rgn,
                              const Parameters& params) {
    for (int i = 0; i < num_rgn; i++) {

        const int num_nb = nbh[i].num_nb;

        double* posteriori_weight = new double[num_nb];
        computePosterioriWeight(nbh, rfeature, posteriori_weight, i, params);


        cv::Vec3d tmp(0, 0, 0);
        for (int k = 0; k < num_nb; k++) {
            tmp += posteriori_weight[k] * nbh[i].nb[k].pl;
        }

        plane[i] = tmp;

        delete[] posteriori_weight;
    }
}

/**
 * Disparity plane refinement using maximum-a-posterior (MAP)
 */
void SurfaceFitting::planeMAP(Neighborhood* nbh, RgnFeature* rfeature,
                              cv::Vec3d* plane, const int& num_rgn,
                              const Parameters& params) {
    for (int i = 0; i < num_rgn; i++) {

        const int num_nb = nbh[i].num_nb;

        double* posteriori_weight = new double[num_nb];
        computePosterioriWeight(nbh, rfeature, posteriori_weight, i, params);


        double max_weight = 0;
        int max_id = 0;
        for (int k = 0; k < num_nb; k++) {
            if (posteriori_weight[k] > max_weight) {
                max_weight = posteriori_weight[k];
                max_id = k;
            }
        }
        plane[i] = nbh[i].nb[max_id].pl;

        delete[] posteriori_weight;
    }
}

/**
 * Assign the plane label to per-pixel disparities
 */
void SurfaceFitting::assignResults(cv::Vec3d* plane, const cv::Mat& rgn,
                                   cv::Mat& labeling, cv::Mat& refined_disp) {
    labeling = cv::Mat(height, width, CV_32FC3);
    refined_disp = cv::Mat(height, width, CV_32FC1);

    for (int x = 0; x < height; x++)
        for (int y = 0; y < width; y++) {
            int tl = rgn.at<int>(x, y);

            labeling.at<cv::Vec3f>(x, y) = plane[tl];

            double disp = plane[tl].ddot(cv::Vec3d(x, y, 1));
            if (disp < 0) disp = 0;
            else if (disp > MAX_DISPARITY) disp = MAX_DISPARITY;
            refined_disp.at<float>(x, y) = disp;
        }
}

/**
 * help function for sorting the configurations
 */
bool SurfaceFitting::Configuration::operator<(Configuration const& c) const {
    if (group_num != c.group_num)
        return(group_num < c.group_num);
    else
        return(sample_num > c.sample_num);
}
