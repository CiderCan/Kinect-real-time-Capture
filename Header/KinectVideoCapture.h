
// KinectVideoCapture.h : PROJECT_NAME ���� ���α׷��� ���� �� ��� �����Դϴ�.
//

#pragma once

#ifndef __AFXWIN_H__
	#error "PCH�� ���� �� ������ �����ϱ� ���� 'stdafx.h'�� �����մϴ�."
#endif



// CKinectVideoCaptureApp:
// �� Ŭ������ ������ ���ؼ��� KinectVideoCapture.cpp�� �����Ͻʽÿ�.
//

class CKinectVideoCaptureApp : public CWinApp
{
public:
	CKinectVideoCaptureApp();

// �������Դϴ�.
public:
	virtual BOOL InitInstance();

// �����Դϴ�.

	DECLARE_MESSAGE_MAP()
};

extern CKinectVideoCaptureApp theApp;