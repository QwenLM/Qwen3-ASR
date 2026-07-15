#pragma once

// Windows-safe UTF-8 command-line handling for the example CLIs.
//
// When a native Windows .exe is launched (e.g. from Git Bash, cmd, or PowerShell),
// argv is handed to the process in the ANSI code page, which mangles non-ASCII
// arguments such as Chinese model paths or prompts. get_utf8_args() recovers the
// real UTF-8 arguments from the wide command line.

#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

// Recover argv as UTF-8. On non-Windows platforms this just copies argv verbatim.
inline std::vector<std::string> get_utf8_args(int argc, char** argv) {
    std::vector<std::string> args;
#ifdef _WIN32
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv) {
        for (int i = 0; i < wargc; i++) {
            int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
            std::string s(n > 0 ? n - 1 : 0, '\0');
            if (n > 1) {
                WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, &s[0], n, nullptr, nullptr);
            }
            args.push_back(std::move(s));
        }
        LocalFree(wargv);
        return args;
    }
#endif
    for (int i = 0; i < argc; i++) args.push_back(argv[i]);
    return args;
}

// Make the Windows console read/write UTF-8 (no-op on a pty such as Git Bash, and on
// non-Windows platforms). Improves Chinese stdin/stdout handling in cmd / PowerShell.
inline void enable_utf8_console() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}
