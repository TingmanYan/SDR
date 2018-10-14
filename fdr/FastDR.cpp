#include "FastDR.h"
#include "../Superpixel/segment-graph.h"
#include "../GCoptimization/GCoptimization.h"

/**
 * Graph-based segmentation of Felzenszwalb and Huttenlocher
 * We have chosen a small k to generate superpixels
 */
void FastDR::graphSegmentation(const cv::Mat I, cv::Mat &spx, int &num_spx) {
    const int min_size = 10;
    const float sigma = 0.1;
    const float k = params.seg_k;

    image<float> *r = new image<float>(width, height);
    image<float> *g = new image<float>(width, height);
    image<float> *b = new image<float>(width, height);

    // smooth each color channel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            imRef(r, x, y) = I.at<cv::Vec3b>(y, x)[2];
            imRef(g, x, y) = I.at<cv::Vec3b>(y, x)[1];
            imRef(b, x, y) = I.at<cv::Vec3b>(y, x)[0];
        }
    }
    image<float> *smooth_r = smooth(r, sigma);
    image<float> *smooth_g = smooth(g, sigma);
    image<float> *smooth_b = smooth(b, sigma);
    delete r;
    delete g;
    delete b;

    // build graph
    edge *edges = new edge[width*height * 4];
    int num = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (x < width - 1) {
                edges[num].a = y * width + x;
                edges[num].b = y * width + (x + 1);
                edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x + 1, y);
                num++;
            }

            if (y < height - 1) {
                edges[num].a = y * width + x;
                edges[num].b = (y + 1) * width + x;
                edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x, y + 1);
                num++;
            }

            if ((x < width - 1) && (y < height - 1)) {
                edges[num].a = y * width + x;
                edges[num].b = (y + 1) * width + (x + 1);
                edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x + 1, y + 1);
                num++;
            }

            if ((x < width - 1) && (y > 0)) {
                edges[num].a = y * width + x;
                edges[num].b = (y - 1) * width + (x + 1);
                edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x + 1, y - 1);
                num++;
            }
        }
    }
    delete smooth_r;
    delete smooth_g;
    delete smooth_b;

    // segment
    universe *u = segment_graph(width*height, num, edges, k);

    //post process small components
    for (int i = 0; i < num; i++) {
        int a = u->find(edges[i].a);
        int b = u->find(edges[i].b);
        if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
            u->join(a, b);
    }
    delete[] edges;
    num_spx = u->num_sets();

    /**
     * The interface has been modified to make the generated label arrange from
     * 0 to num_spx - 1
     */
    int* L = new int[width*height]{ 0 };
    int* H = new int[width*height]{ 0 };

    for (int p = 0; p < width*height; p++) L[p] = u->find(p);
    for (int p = 0; p < width*height; p++) H[L[p]]++;

    int num_L = 0;
    for (int p = 0; p < width*height; p++) if (H[p]) H[p] = num_L++;
    num_spx = num_L;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int comp = H[L[y * width + x]];
            spx.at<int>(y, x) = comp;
        }
    }

    delete[] H;
    delete[] L;
    delete u;
}

/**
 * Compute the features of superpixels,
 * which include minimum surrounding box, color, and number of pixels
 */
void FastDR::computeSpxFeature(const cv::Mat I, const cv::Mat spx,
                               const int num_spx, SpxFeature *sfeature) {
    for (int k = 0; k < num_spx; k++) {
        sfeature[k].tl = cv::Point2i(height, width);
        sfeature[k].br = cv::Point2i(0, 0);
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int tl = spx.at<int>(i, j);

            sfeature[tl].br.x = i > sfeature[tl].br.x ? i : sfeature[tl].br.x;
            sfeature[tl].br.y = j > sfeature[tl].br.y ? j : sfeature[tl].br.y;
            sfeature[tl].tl.x = i < sfeature[tl].tl.x ? i : sfeature[tl].tl.x;
            sfeature[tl].tl.y = j < sfeature[tl].tl.y ? j : sfeature[tl].tl.y;

            sfeature[tl].color = (sfeature[tl].color * sfeature[tl].num_pixel +
                                  (cv::Vec3f)I.at<cv::Vec3b>(i, j)) /
                                 (sfeature[tl].num_pixel + 1);
            sfeature[tl].num_pixel++;
        }
    }
}

/**
 * Adjacent matrix
 * adj_mat[i][j] denote whether two superpixels i and j are adjacent
 * also the edge length of a superpixel is computed
 */
void FastDR::computeAdjmat(const cv::Mat spx, uchar **adj_mat,
                           int **edge_length) {
    const int dx8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
    const int dy8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };

    int cl, tl;
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++) {
            cl = spx.at<int>(i, j);
            for (int k = 0; k < 8; k++) {
                int x = i + dx8[k];
                int y = j + dy8[k];
                if ((x >= 0 && x < height) && (y >= 0 && y < width)) {
                    tl = spx.at<int>(x, y);
                    if (tl != cl) {
                        adj_mat[tl][cl] = 1;
                        adj_mat[cl][tl] = 1;
                        edge_length[tl][cl] += 1;
                        edge_length[cl][tl] += 1;
                        break;
                    }
                }
            }
        }
}

/**
 * Compute the data term,
 * which is histograms of disparity distribution in a superpixel
 */
void FastDR::computeDataTerm(const cv::Mat D, const cv::Mat spx, const int
num_spx, SpxFeature * sfeature, float ** data_term, const int bins_width)
{
    // in case of bugs
    for (int k = 0; k < num_spx; k++)
        if (sfeature[k].num_pixel == 0)
            sfeature[k].num_pixel = 1;

    int tl;
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++) {
            tl = spx.at<int>(i, j);
            int disp = D.at<float>(i, j);
            data_term[tl][disp / bins_width] += 1;
        }

    const int num_bins = (MAX_DISPARITY - 1) / bins_width + 1;
    for (int k = 0; k < num_spx; k++) {
        int num_pixel = sfeature[k].num_pixel;
        for (int t = 0; t < num_bins; t++) {
            data_term[k][t] = num_pixel - data_term[k][t];
        }
    }
}

/**
 * Graph-Cut optimization to optimize the MRF
 * The first expansion then swap moves are used here
 */
void FastDR::graphCutOptimization(float ** data_term, uchar ** adj_mat, int **
edge_length, SpxFeature * sfeature, const int num_bins, const int num_spx, const
Parameters params)
{
    const int num_vertices = num_spx;
    const int num_labels = num_bins;
    const double lambda = params.lambda;
    const double gamma = params.gamma;
    const int tau = params.tau;

    // gc -> setNeighbors(int i, int j, int weight)
    // casting the weight from double to int will lose information
    // enlarge the data term and smoothness term simultaneously
    const int scale_factor = 2;

    // first set up the array for data costs
    int *data = new int[num_vertices*num_labels];
    int *smooth = new int[num_labels*num_labels];
    for (int i = 0; i < num_vertices; i++)
        for (int l = 0; l < num_labels; l++) {
            data[i * num_labels + l] = data_term[i][l] * scale_factor;
        }

    // alpha expansion 
    for (int l1 = 0; l1 < num_labels; l1++)
        for (int l2 = 0; l2 < num_labels; l2++)
            smooth[l1 + l2*num_labels] = abs(l1 - l2);
    try {
        GCoptimizationGeneralGraph *gc = new
GCoptimizationGeneralGraph(num_vertices, num_labels);

        gc->setDataCost(data);
        gc->setSmoothCost(smooth);

        //now set up a neighborhood system
        for (int i = 0; i <num_vertices; i++)
            for (int j = i + 1; j < num_vertices; j++)
                if (adj_mat[i][j]) {
                    cv::Vec3f colordiff = sfeature[i].color - sfeature[j].color;
                    double omega_ij = sqrt(colordiff.ddot(colordiff));
                    omega_ij = lambda * std::max(exp(-omega_ij / gamma), 0.01) *
                               edge_length[i][j] * scale_factor;
                    gc->setNeighbors(i, j, omega_ij);
                }

        gc->expansion(-1);  // perform expansion moves till converge
        // gc->expansion(2) two iterations of expansion moves

        for (int k = 0; k < num_vertices; k++)
            sfeature[k].mean = gc->whatLabel(k);

        delete gc;
    }
    catch (GCException e) {
        e.Report();
    }

    // alpha-beta swap
    if (params.use_swap) {
        for (int l1 = 0; l1 < num_labels; l1++)
            for (int l2 = 0; l2 < num_labels; l2++) {
                if (l2 == l1) smooth[l1 + l2*num_labels] = 0;
                if (abs(l2 - l1) == 1) smooth[l1 + l2*num_labels] = 1;
                if (abs(l2 - l1) > 1) smooth[l1 + l2*num_labels] = tau;
            }

        try {
            GCoptimizationGeneralGraph *gc =
                new GCoptimizationGeneralGraph(num_vertices, num_labels);

            for (int k = 0; k < num_vertices; k++)
                gc->setLabel(k, sfeature[k].mean);

            gc->setDataCost(data);
            gc->setSmoothCost(smooth);

            for (int i = 0; i < num_vertices; i++)
                for (int j = i + 1; j < num_vertices; j++)
                    if (adj_mat[i][j]) {
                        cv::Vec3f colordiff =
                            sfeature[i].color - sfeature[j].color;
                        double omega_ij = sqrt(colordiff.ddot(colordiff));
                        omega_ij = lambda *
                                   std::max(exp(-omega_ij / gamma), 0.01) *
                                   edge_length[i][j] * scale_factor;
                        gc->setNeighbors(i, j, omega_ij);
                    }

            gc->swap(-1);

            for (int k = 0; k < num_vertices; k++)
                sfeature[k].mean = gc->whatLabel(k);

            delete gc;
        }
        catch (GCException e) {
            e.Report();
        }
    }

    for (int k = 0; k < num_spx; k++)
        sfeature[k].disp = sfeature[k].mean * params.bins_width;

    delete[] smooth;
    delete[] data;
}

/**
 * Adaptive mean filter
 */
void FastDR::adaptiveMeanFilter(cv::Mat & D)
{
    // get disparity image dimensions
    int32_t D_width = D.cols;
    int32_t D_height = D.rows;

    // allocate temporary memory
    cv::Mat D_copy = D.clone();
    cv::Mat D_tmp(D.rows, D.cols, CV_32FC1, cv::Scalar::all(0));

    // zero input disparity maps to -10 (this makes the bilateral
    // weights of all valid disparities to 0 in this region)
    for (int32_t u = 0; u < D_width; u++) {
        for (int32_t v = 0; v < D_height; v++) {
            if (D.at<float>(v, u) < 0) {
                D_copy.at<float>(v, u) = -10;
                D_tmp.at<float>(v, u) = -10;
            }
        }
    }

    __m128 xconst0 = _mm_set1_ps(0);
    __m128 xconst4 = _mm_set1_ps(4);
    __m128 xval, xweight1, xweight2, xfactor1, xfactor2;

    float *val = (float *)_mm_malloc(8 * sizeof(float), 16);
    float *weight = (float*)_mm_malloc(4 * sizeof(float), 16);
    float *factor = (float*)_mm_malloc(4 * sizeof(float), 16);

    // set absolute mask
    __m128 xabsmask = _mm_set1_ps(0x7FFFFFFF);


    // horizontal filter
    for (int32_t v = 3; v < D_height - 3; v++) {

        // init
        for (int32_t u = 0; u < 7; u++)
            val[u] = D_copy.at<float>(v, u);

        // loop
        for (int32_t u = 7; u < D_width; u++) {

            // set
            float val_curr = D_copy.at<float>(v, u - 3);
            val[u % 8] = D_copy.at<float>(v, u);

            xval = _mm_load_ps(val);
            xweight1 = _mm_sub_ps(xval, _mm_set1_ps(val_curr));
            xweight1 = _mm_and_ps(xweight1, xabsmask);
            xweight1 = _mm_sub_ps(xconst4, xweight1);
            xweight1 = _mm_max_ps(xconst0, xweight1);
            xfactor1 = _mm_mul_ps(xval, xweight1);

            xval = _mm_load_ps(val + 4);
            xweight2 = _mm_sub_ps(xval, _mm_set1_ps(val_curr));
            xweight2 = _mm_and_ps(xweight2, xabsmask);
            xweight2 = _mm_sub_ps(xconst4, xweight2);
            xweight2 = _mm_max_ps(xconst0, xweight2);
            xfactor2 = _mm_mul_ps(xval, xweight2);

            xweight1 = _mm_add_ps(xweight1, xweight2);
            xfactor1 = _mm_add_ps(xfactor1, xfactor2);

            _mm_store_ps(weight, xweight1);
            _mm_store_ps(factor, xfactor1);

            float weight_sum = weight[0] + weight[1] + weight[2] + weight[3];
            float factor_sum = factor[0] + factor[1] + factor[2] + factor[3];

            if (weight_sum > 0) {
                float d = factor_sum / weight_sum;
                if (d >= 0) D_tmp.at<float>(v, u - 3) = d;
            }
        }
    }

    // vertical filter
    for (int32_t u = 3; u < D_width - 3; u++) {

        // init
        for (int32_t v = 0; v < 7; v++)
            val[v] = D_tmp.at<float>(v, u);

        // loop
        for (int32_t v = 7; v < D_height; v++) {

            // set
            float val_curr = D_tmp.at<float>(v - 3, u);
            val[v % 8] = D_tmp.at<float>(v, u);

            xval = _mm_load_ps(val);
            xweight1 = _mm_sub_ps(xval, _mm_set1_ps(val_curr));
            xweight1 = _mm_and_ps(xweight1, xabsmask);
            xweight1 = _mm_sub_ps(xconst4, xweight1);
            xweight1 = _mm_max_ps(xconst0, xweight1);
            xfactor1 = _mm_mul_ps(xval, xweight1);

            xval = _mm_load_ps(val + 4);
            xweight2 = _mm_sub_ps(xval, _mm_set1_ps(val_curr));
            xweight2 = _mm_and_ps(xweight2, xabsmask);
            xweight2 = _mm_sub_ps(xconst4, xweight2);
            xweight2 = _mm_max_ps(xconst0, xweight2);
            xfactor2 = _mm_mul_ps(xval, xweight2);

            xweight1 = _mm_add_ps(xweight1, xweight2);
            xfactor1 = _mm_add_ps(xfactor1, xfactor2);

            _mm_store_ps(weight, xweight1);
            _mm_store_ps(factor, xfactor1);

            float weight_sum = weight[0] + weight[1] + weight[2] + weight[3];
            float factor_sum = factor[0] + factor[1] + factor[2] + factor[3];

            if (weight_sum > 0) {
                float d = factor_sum / weight_sum;
                if (d >= 0) D.at<float>(v - 3, u);
            }
        }
    }


    // free memory
    _mm_free(val);
    _mm_free(weight);
    _mm_free(factor);
}

/**
 * Median filter
 */
void FastDR::medianFilter(cv::Mat & D)
{
    // get disparity image dimensions
    int32_t D_width = D.cols;
    int32_t D_height = D.rows;

    // temporary memory
    cv::Mat D_temp(D.rows, D.cols, CV_32FC1, cv::Scalar::all(0));
    int32_t window_size = 3;

    float *vals = new float[window_size * 2 + 1];
    int32_t i, j;
    float temp;

    // first step: horizontal median filter
    for (int32_t u = window_size; u<D_width - window_size; u++) {
        for (int32_t v = window_size; v<D_height - window_size; v++) {
            if (D.at<float>(v, u) >= 0) {
                j = 0;
                for (int32_t u2 = u - window_size; u2 <= u + window_size; u2++) {
                    temp = D.at<float>(v, u2);
                    i = j - 1;
                    while (i >= 0 && *(vals + i)>temp) {
                        *(vals + i + 1) = *(vals + i);
                        i--;
                    }
                    *(vals + i + 1) = temp;
                    j++;
                }
                D_temp.at<float>(v, u) = *(vals + window_size);
            }
            else {
                D_temp.at<float>(v, u) = D.at<float>(v, u);
            }

        }
    }

    // second step: vertical median filter
    for (int32_t u = window_size; u<D_width - window_size; u++) {
        for (int32_t v = window_size; v<D_height - window_size; v++) {
            if (D.at<float>(v, u) >= 0) {
                j = 0;
                for (int32_t v2 = v - window_size; v2 <= v + window_size; v2++) {
                    temp = D_temp.at<float>(v2, u);
                    i = j - 1;
                    while (i >= 0 && *(vals + i)>temp) {
                        *(vals + i + 1) = *(vals + i);
                        i--;
                    }
                    *(vals + i + 1) = temp;
                    j++;
                }
                D.at<float>(v, u) = *(vals + window_size);
            }
            else {
                D.at<float>(v, u) = D.at<float>(v, u);
            }
        }
    }

    delete[] vals;
}
