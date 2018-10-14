/********************************************************************
 * This code is copyed from taniai
 * Interface functions to read and write pfm files and cost volumes
 *******************************************************************/
#pragma once

#include <opencv2/opencv.hpp>
#include <fstream>

namespace cvutils
{
	namespace io
	{
		using byte = uchar;
		static int is_little_endian()
		{
			if (sizeof(float) != 4)
			{
				printf("Bad float size.\n"); exit(1);
			}
			byte b[4] = { 255, 0, 0, 0 };
			return *((float *)b) < 1.0;
		}

		static cv::Mat read_pfm_file(const std::string& filename)
		{
			int w, h;
			char buf[256];
            FILE *f=fopen(filename.c_str(),"rb");
			if (f == NULL)
			{
				//wprintf(L"PFM file absent: %s\n\n", filename.c_str());
				return cv::Mat();
			}

			int channel = 1;
			fscanf(f, "%s\n", buf);
			if (strcmp(buf, "Pf") == 0) channel = 1;
			else if (strcmp(buf, "PF") == 0) channel = 3;
			else {
				printf(buf);
				printf("Not a 1/3 channel PFM file.\n");
				return cv::Mat();
			}
			fscanf(f, "%d %d\n", &w, &h);
			double scale = 1.0;
			fscanf(f, "%lf\n", &scale);
			int little_endian = 0;
			if (scale < 0.0)
			{
				little_endian = 1;
				scale = -scale;
			}
			size_t datasize = w*h*channel;
			std::vector<byte> data(datasize * sizeof(float));

			cv::Mat image = cv::Mat(h, w, CV_MAKE_TYPE(CV_32F, channel));

			// Adjust the position of the file because fscanf() reads too much (due to "\n"?)
			fseek(f, -(long)datasize * sizeof(float), SEEK_END);
			size_t count = fread((void *)&data[0], sizeof(float), datasize, f);
			if (count != datasize)
			{
				printf("Expected size %d, read size %d.\n", datasize, count);
				printf("Could not read ground truth file.\n");
				return cv::Mat();
			}
			int native_little_endian = is_little_endian();
            for (size_t i = 0; i < datasize; i++) {
                byte* p = &data[i * 4];
                if (little_endian != native_little_endian) {
                    byte temp;
                    temp = p[0];
                    p[0] = p[3];
                    p[3] = temp;
                    temp = p[1];
                    p[1] = p[2];
                    p[2] = temp;
                }
                int jj = (i / channel) % w;
                int ii = (i / channel) / w;
                int ch = i % channel;
                image.at<float>(h - 1 - ii, jj * channel + ch) =
                    *((float*)p);
            }
            fclose(f);
			return image;
		}

		static void save_pfm_file(const std::string& filename, const cv::Mat& image)
		{
			int width = image.cols;
			int height = image.rows;

            FILE *stream=fopen(filename.c_str(),"wb");
			if (stream == NULL)
			{
				wprintf(L"PFM file absent: %s\n\n", filename.c_str());
				return;
			}
			// write the header: 3 lines: Pf, dimensions, scale factor (negative val == little endian)
			int channel = image.channels();
			if (channel == 1)
				fprintf(stream, "Pf\n%d %d\n%lf\n", width, height, -1.0 / 255.0);
			else if (channel == 3)
				fprintf(stream, "PF\n%d %d\n%lf\n", width, height, -1.0 / 255.0);
			else {
				printf("Channels %d must be 1 or 3\n", image.channels());
				return;
			}


			// pfm stores rows in inverse order!
			int linesize = width*channel;
			std::vector<float> rowBuff(linesize);
			for (int y = height - 1; y >= 0; y--)
			{
				auto ptr = image.ptr<float>(y);
				auto pBuf = &rowBuff[0];
				for (int x = 0; x < linesize; x++)
				{
					float val = (float)(*ptr);
					pBuf[x] = val;
					ptr++;
					/*if (val > 0 && val <= 255)
					rowBuf[x] = val;
					else
					{
					printf("invalid: val %f\n", flo(x,y));
					rowBuf[x] = 0.0f;
					}*/
				}
				if ((int)fwrite(&rowBuff[0], sizeof(float), width, stream) != width)
				{
					printf("[ERROR] problem with fwrite.");
				}
				fflush(stream);
			}

			fclose(stream);
			return;
		}


		static bool writeMatBinary(std::ofstream& ofs, const cv::Mat& out_mat)
		{
			cv::Mat out = out_mat;
			if (!out.isContinuous())
				out = out.clone();

			if (!ofs.is_open()) {
				return false;
			}
			if (out.empty()) {
				int s = 0;
				ofs.write((const char*)(&s), sizeof(int));
				return true;
			}
			int rows = out.rows;
			int cols = out.cols;
			int type = out.type();

			ofs.write((const char*)(&rows), sizeof(int));
			ofs.write((const char*)(&cols), sizeof(int));
			ofs.write((const char*)(&type), sizeof(int));
			ofs.write((const char*)(out.data), out.elemSize() * out.total());

			return true;
		}


		static bool saveMatBinary(const std::string& filename, const cv::Mat& output) {
			std::ofstream ofs(filename, std::ios::binary);
			return writeMatBinary(ofs, output);
		}


		static bool readMatBinary(std::ifstream& ifs, cv::Mat& in_mat, bool readHeader = true)
		{
			if (!ifs.is_open()) {
				return false;
			}

			if (readHeader)
			{
				int rows, cols, type;
				ifs.read((char*)(&rows), sizeof(int));
				if (rows == 0) {
					return true;
				}
				ifs.read((char*)(&cols), sizeof(int));
				ifs.read((char*)(&type), sizeof(int));

				in_mat.release();
				in_mat.create(rows, cols, type);
			}
			ifs.read((char*)(in_mat.data), in_mat.elemSize() * in_mat.total());

			return true;
		}


		static bool loadMatBinary(const std::string& filename, cv::Mat& output, bool readHeader = true) {
			std::ifstream ifs(filename, std::ios::binary);
			return readMatBinary(ifs, output, readHeader);
		}

	}
	inline bool contains(const std::string& str1, const std::string& str2)
	{
		std::string::size_type pos = str1.find(str2);
		if (pos == std::string::npos)
		{
			return false;
		}
		return true;
	}
}  // namespace cvutils
