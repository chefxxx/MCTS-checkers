//
// Created by chefxx on 8.01.2026.
//

#ifndef LOGGER_H
#define LOGGER_H

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace logger {

struct Format
{
    std::string_view str;

    template <size_t N>
    // NOLINTNEXTLINE(google-explicit-constructor)
    consteval Format(const char (&t_s)[N])
        : str(t_s, N - 1)
    {
        parse_and_validate();
    }

private:
    consteval void parse_and_validate() const
    {
        for (size_t i = 0; i < str.size(); ++i) {
            if (str[i] == '{' && (i + 1 == str.size() || str[i + 1] != '}')) {
                throw "{} mismatch!";
            }
        }
    }
};

namespace helpers {

inline void helper(const std::string_view t_str)
{
    const auto now      = std::chrono::system_clock::now();
    const auto now_time = std::chrono::system_clock::to_time_t(now);

    // TODO::localtime is not thread-safe; localtime_r or localtime_s is preferred in production
    const auto local_tm = *std::localtime(&now_time);

    std::cout << "[" << t_str << "][" << std::put_time(&local_tm, "%H:%M:%S") << "] ";
}

inline void print_recursive(const std::string_view t_str) { std::cout << t_str; }

template <typename First, typename... Rest>
void print_recursive(std::string_view t_str, First &&t_first, Rest &&...t_rest)
{
    const size_t pos = t_str.find("{}");
    if (pos == std::string_view::npos) {
        print_recursive(t_str);
        return;
    }

    std::cout << t_str.substr(0, pos);
    std::cout << std::forward<First>(t_first);

    print_recursive(t_str.substr(pos + 2), std::forward<Rest>(t_rest)...);
}

} // namespace helpers

inline void info(const Format t_formatWrapper)
{
    helpers::helper("INFO");
    std::cout << t_formatWrapper.str;
}

template <typename... Args> void info(const Format t_formatWrapper, Args &&...t_arguments)
{
    helpers::helper("INFO");
    helpers::print_recursive(t_formatWrapper.str, std::forward<Args>(t_arguments)...);
}

inline void warn(const Format t_formatWrapper)
{
    helpers::helper("WARN");
    std::cout << t_formatWrapper.str;
}

template <typename... Args> void warn(const Format t_formatWrapper, Args &&...t_arguments)
{
    helpers::helper("WARN");
    helpers::print_recursive(t_formatWrapper.str, std::forward<Args>(t_arguments)...);
}

inline void err(const Format t_formatWrapper)
{
    helpers::helper("ERR");
    std::cout << t_formatWrapper.str;
}

template <typename... Args> void err(const Format t_formatWrapper, Args &&...t_arguments)
{
    helpers::helper("ERR");
    helpers::print_recursive(t_formatWrapper.str, std::forward<Args>(t_arguments)...);
}

} // namespace logger

#endif // LOGGER_H
