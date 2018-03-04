
// stdafx.h : 자주 사용하지만 자주 변경되지는 않는
// 표준 시스템 포함 파일 및 프로젝트 관련 포함 파일이 
// 들어 있는 포함 파일입니다.

#pragma once

#ifndef VC_EXTRALEAN
#define VC_EXTRALEAN            // 거의 사용되지 않는 내용은 Windows 헤더에서 제외합니다.
#endif

#pragma comment(linker, "/entry:WinMainCRTStartup /subsystem:console")

#define _CRT_SECURE_CPP_OVERLOAD_SECURE_NAMES 1

#include "targetver.h"

#define _ATL_CSTRING_EXPLICIT_CONSTRUCTORS      // 일부 CString 생성자는 명시적으로 선언됩니다.

// MFC의 공통 부분과 무시 가능한 경고 메시지에 대한 숨기기를 해제합니다.
#define _AFX_ALL_WARNINGS

#include <afxwin.h>         // MFC 핵심 및 표준 구성 요소입니다.
#include <afxext.h>         // MFC 확장입니다.

#include <new>
#include <windows.h>
#include <strsafe.h>
#include <string>
#include <afxdisp.h>        // MFC 자동화 클래스입니다.



#ifndef _AFX_NO_OLE_SUPPORT
#include <afxdtctl.h>           // Internet Explorer 4 공용 컨트롤에 대한 MFC 지원입니다.
#endif
#ifndef _AFX_NO_AFXCMN_SUPPORT
#include <afxcmn.h>             // Windows 공용 컨트롤에 대한 MFC 지원입니다.
#endif // _AFX_NO_AFXCMN_SUPPORT

#include <afxcontrolbars.h>     // MFC의 리본 및 컨트롤 막대 지원
#include <afxsock.h> // MFC socket lib


// Case-insensitive wide-character char traits
struct i_char_traits : public std::char_traits<wchar_t>
{
	static bool eq(wchar_t c1, wchar_t c2) { return towupper(c1) == towupper(c2); }
	static bool ne(wchar_t c1, wchar_t c2) { return towupper(c1) != towupper(c2); }
	static bool lt(wchar_t c1, wchar_t c2) { return towupper(c1) <  towupper(c2); }
	static int compare(const wchar_t* s1, const wchar_t* s2, size_t n)
	{
		while (0 != n--)
		{
			if (towupper(*s1) < towupper(*s2)) return -1;
			if (towupper(*s1) > towupper(*s2)) return 1;
			++s1;
			++s2;
		}

		return 0;
	}

	static const wchar_t* find(const wchar_t* s, size_t n, wchar_t a)
	{
		int remaining = static_cast<int>(n);
		while ((--remaining >= 0) && (towupper(*s) != towupper(a)))
		{
			++s;
		}
		return (remaining >= 0) ? s : NULL;
	}
};

typedef std::basic_string<wchar_t, i_char_traits> wistring;


#ifdef _UNICODE
#if defined _M_IX86
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='x86' publicKeyToken='6595b64144ccf1df' language='*'\"")
#elif defined _M_X64
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='amd64' publicKeyToken='6595b64144ccf1df' language='*'\"")
#else
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
#endif
#endif


