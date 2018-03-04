#pragma once
#include "afxwin.h"
#include "opencv.hpp"
#include "app.h"
#include "highgui.hpp"
#include "cv.h"
#include "KinectVideoCaptureDlg.h"
#include <string.h>
#include <fstream>
// CRecordDlg 대화 상자입니다.


// RecordHelpDlg 대화 상자입니다.

class RecordHelpDlg : public CDialogEx
{
	DECLARE_DYNAMIC(RecordHelpDlg)

public:
	RecordHelpDlg(CWnd* pParent = NULL);   // 표준 생성자입니다.
	virtual ~RecordHelpDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_RECORD };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

	DECLARE_MESSAGE_MAP()

public:
	CListBox m_place;
	CListBox m_stuff;
	virtual BOOL OnInitDialog();
	afx_msg void OnDestroy();

	void saveBehavior();
	afx_msg void OnBnClickedOkButton();
	int m_sex;

	std::string stuffStr = "";
	std::string placeStr = "";
	std::string age = "";
	std::string sex = "";
	std::string behavior = "";
};
