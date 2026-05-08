#pragma once

#include <format>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <fe/assert.h>
#include <fe/format.h>

namespace mim {

class Def;

using fe::StreamFn;
using fe::Tab;

/// @name Formatted Output
/// @anchor fmt
/// Wrappers around `std::format` and `std::print` that match this project's API surface.
/// Use `{}` placeholders as in `std::format`.
/// To inject ad-hoc streaming code, wrap a callable into a `fe::StreamFn`.
/// To join range elements with a custom separator, use `fe::join(range, sep)`.
///@{

// clang-format off
template<class... Args> std::string   fmt    (std::format_string<Args...> fmt, Args&&... args) { return std::format(fmt, std::forward<Args>(args)...); }
// clang-format on

/// Wraps `std::format` to throw `T` with a formatted message.
template<class T = std::logic_error, class... Args>
[[noreturn]] void error(std::format_string<Args...> fmt, Args&&... args) {
    throw T("error: " + std::format(fmt, std::forward<Args>(args)...));
}

#ifdef NDEBUG
#    define assertf(condition, ...)  \
        do {                         \
            (void)sizeof(condition); \
        } while (false)
#else
#    define assertf(condition, ...)                                  \
        do {                                                         \
            if (!(condition)) {                                      \
                mim::errf("{}:{}: assertion: ", __FILE__, __LINE__); \
                mim::errln(__VA_ARGS__);                             \
                fe::breakpoint();                                    \
            }                                                        \
        } while (false)
#endif
///@}

/// @name out/err
/// mim::print%s to `std::cout`/`std::cerr`; the *`ln` variants emit an additional `std::endl`.
///@{
// clang-format off
template<class... Args> std::ostream& outf (std::format_string<Args...> fmt, Args&&... args) { return std::cout << std::format(fmt, std::forward<Args>(args)...);                }
template<class... Args> std::ostream& errf (std::format_string<Args...> fmt, Args&&... args) { return std::cerr << std::format(fmt, std::forward<Args>(args)...);                }
template<class... Args> std::ostream& outln(std::format_string<Args...> fmt, Args&&... args) { return std::cout << std::format(fmt, std::forward<Args>(args)...) << std::endl; }
template<class... Args> std::ostream& errln(std::format_string<Args...> fmt, Args&&... args) { return std::cerr << std::format(fmt, std::forward<Args>(args)...) << std::endl; }
// clang-format on
///@}

} // namespace mim

#ifndef DOXYGEN
/// Format any pointer to a `mim::Def` (or subclass) via its `operator<<`.
template<class T>
requires std::derived_from<T, mim::Def>
struct std::formatter<T*> : fe::ostream_formatter {};
template<class T>
requires std::derived_from<T, mim::Def>
struct std::formatter<const T*> : fe::ostream_formatter {};
#endif
