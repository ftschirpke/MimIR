#include "mim/util/sys.h"

#include <array>
#include <filesystem>
#include <iostream>
#include <vector>

#include "mim/util/dbg.h"

#ifdef _WIN32
#    include <windows.h>
#    define popen  _popen
#    define pclose _pclose
#    define WEXITSTATUS
#else
#    include <dlfcn.h>
#endif

using namespace std::string_literals;

namespace mim::sys {

std::optional<fs::path> path_to_curr_libmim() {
#if defined(_WIN32)
    HMODULE mod = nullptr;
    auto flags  = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    if (!GetModuleHandleExW(flags, reinterpret_cast<LPCWSTR>(&path_to_curr_libmim), &mod)) return {};

    wchar_t buf[MAX_PATH];
    DWORD len = GetModuleFileNameW(mod, buf, MAX_PATH);
    if (len == 0) return {};

    return fs::weakly_canonical(fs::path(buf));
#elif defined(__APPLE__) || defined(__linux__)
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&path_to_curr_libmim), &info) == 0) return {};

    if (!info.dli_fname) return {};

    return fs::weakly_canonical(fs::path(info.dli_fname));
#else
    return {};
#endif
}

// see https://stackoverflow.com/a/478960
std::string exec(std::string cmd) {
    using PipeCloser = int (*)(FILE*); // spell out type explicitly to get rid of warning
    if (auto pipe = std::unique_ptr<FILE, PipeCloser>(popen(cmd.c_str(), "r"), pclose)) {
        std::array<char, 128> buffer;
        std::string result;
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
            result += buffer.data();
        return result;
    } else
        error("popen() failed!");
}

std::string find_cmd(std::string cmd) {
    auto out = exec(MIM_WHICH " "s + cmd);
    if (auto it = out.find('\n'); it != std::string::npos) out.erase(it);
    return out;
}

int system(std::string cmd) {
    std::cout << cmd << std::endl;
    int status = std::system(cmd.c_str());
    return WEXITSTATUS(status);
}

int run(std::string cmd, std::string args /* = {}*/) {
#ifdef _WIN32
    cmd += ".exe";
#else
    cmd = "./"s + cmd;
#endif
    return sys::system(cmd + " "s + args);
}

std::string escape(const std::filesystem::path& path) {
    std::string str;
    for (char c : path.string()) {
        if (isspace(c)) str += '\\';
        str += c;
    }
    return str;
}

} // namespace mim::sys
