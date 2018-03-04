
// KinectVideoCaptureDlg.h : 헤더 파일
//

#pragma once
#include "afxwin.h"
#include "opencv.hpp"
#include "app.h"
#include "highgui.hpp"
#include "cv.h"
#include <fstream>
using namespace cv;

typedef struct {
	int width;
	int height;
	int imageSize;
	char* pImageData;

	int dWidth;
	int dHeight;
	int dImageSize;
	char* dpImageData;

	BODY_POINT * pBodyPoint;
}RECODE_FORMAT;


// CKinectVideoCaptureDlg 대화 상자
class CKinectVideoCaptureDlg : public CDialogEx
{
// 생성입니다.
public:
	CKinectVideoCaptureDlg(CWnd* pParent = NULL);	// 표준 생성자입니다.

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_KINECTVIDEOCAPTURE_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.


// 구현입니다.
protected:
	HICON m_hIcon;

	// 생성된 메시지 맵 함수
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	//BUFFER
	CList<RECODE_FORMAT, RECODE_FORMAT>  m_RecodFrameList;
	CCriticalSection m_ctRecode;

	RECT r;
	RECT r2;

	CStatic m_picture;
	CStatic m_picture2;

	afx_msg void OnDestroy();

	Kinect dKinect;
	Kinect cKinect;
	Kinect bKinect;

	Mat dmat_frame;
	Mat cmat_frame;
	BODY_POINT	m_bodyPoint[6 * 25];

	Mat bg;
	Ptr<BackgroundSubtractorMOG2> pMog2;

	CImage *cimage_mfc;
	CImage *cimage2_mfc;

	CString str;

	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedButton3();
	void getTime();
	int getRate();

	CString mDay;
	CString mTime;
	CString valDrawRate, valPartRate, valStartTime, valEndTime, valSaveRoute;
	std::string saveRoute;

	int wCount;
	afx_msg void OnBnClickedButton2();
	CButton autoBtn;

	bool isAutoRecord;
	bool isRecording;

	afx_msg void OnBnClickedPortButton();

	CSocket servSock;
	CSocket clintSock;
	UINT m_strPort;
	std::ofstream depthRaw;
	std::ofstream m_oBodyPoint;
	
	LARGE_INTEGER liCounter1, liCounter2, liFrequency;
	void inQFrame();

	int drawTime;
	int drawRate;
	int partRate;
	int partTime;
	time_t start;
	time_t end;
	void tic();
	double toc();
	int fps;
	int frameNum;
	double nDelay;
	int startTime;
	int endTime;

	int m_iRecod;
	afx_msg void OnTimer(UINT_PTR nIDEvent);

	void RecordStart(std::string stuffStr, std::string placeStr, std::string age, std::string sex, std::string behavior, std::string part);
	void RecordStart(std::string time, std::string part);
	void RecordEnd();

	std::string stuffStr;
	std::string placeStr;
	std::string age;
	std::string sex;
	std::string behavior;
	int part;

	Size winSize;
	Size winSize2;

	bool sleepTime;
};