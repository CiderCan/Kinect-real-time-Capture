#pragma once
#ifndef __APP__
#define __APP__

#include "stdafx.h"
#include <Windows.h>
#include <Kinect.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <array>

#include <wrl/client.h>
using namespace Microsoft::WRL;

#pragma pack(push,1)
typedef struct _BODY_POINT {
	int x;
	int y;
	unsigned short z;
} BODY_POINT;
#pragma pack(pop)

class Kinect
{
private:
    // Sensor
    ComPtr<IKinectSensor> kinect;

	// Coordinate Mapper
	ComPtr<ICoordinateMapper> coordinateMapper;

    // Reader
    ComPtr<IColorFrameReader> colorFrameReader;
	ComPtr<IDepthFrameReader> depthFrameReader;
	ComPtr<IBodyFrameReader> bodyFrameReader;

    // Color 
    std::vector<BYTE> colorBuffer;
    int colorWidth;
    int colorHeight;
    unsigned int colorBytesPerPixel;

	//Depth Buffer
	std::vector<UINT16> depthBuffer;
	int depthWidth;
	int depthHeight;

	unsigned int depthBytesPerPixel;
	bool m_bInit;
public:
	cv::Mat colorMat;
	cv::Mat depthMat;
	cv::VideoWriter outputVideo;

	// Body Buffer
	std::array<IBody*, BODY_COUNT> bodies = { nullptr };
	std::array<cv::Vec3b, BODY_COUNT> colors;

	// Constructor
	Kinect();
	void initialize(int);

    // Destructor
    ~Kinect();

    // Initialize Sensor
    inline void initializeSensor();
    inline void initializeColor();
	inline void initializeDepth();
	inline void initializeBody();

    // Finalize
    void finalize();

    // Update Color
    bool updateColor();
	bool updateDepth();

	void drawColor();
	void drawDepth();
	bool updateBodyDraw(cv::Mat& mat, IplImage* pOrg, BODY_POINT* pPoint);
	bool updateBody(IplImage* pOrg, BODY_POINT* pPoint);

	inline void drawEllipse(cv::Mat& image, const Joint& joint, const int radius, const cv::Vec3b& color, const int thickness = -1);

	// Draw Hand State
	inline void drawHandState(cv::Mat& image, const Joint& joint, HandState handState, TrackingConfidence handConfidence);

	inline void GetBodyPoint(IplImage* pOrg, const Joint& joint, BODY_POINT* pBody);

};

#endif // __APP__