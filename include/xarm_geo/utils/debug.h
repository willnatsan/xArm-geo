#pragma once

#include <iostream>
#include <source_location>
#include <string_view>

namespace xarm_geo::debug {

#ifdef NDEBUG
    inline constexpr bool kEnabled = false;
#else
    inline constexpr bool kEnabled = true;
#endif

    // Debug-only diagnostic. In release builds (NDEBUG) the body is
    // discarded by `if constexpr`, leaving an empty inline function the
    // optimiser can elide entirely.
    //
    // Sink is std::cerr today; centralised here so it can be swapped
    // (spdlog, fmt, capture-for-tests, ...) without touching call sites.
    inline void log(std::string_view msg,
                    std::source_location loc = std::source_location::current()) {
        if constexpr (kEnabled) {
            std::cerr << "[xarm_geo " << loc.function_name() << "] " << msg << '\n';
        }
    }

}  // namespace xarm_geo::debug
