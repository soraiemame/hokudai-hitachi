#ifndef HEADER_2HC2022_ERROR_CHECK
#define HEADER_2HC2022_ERROR_CHECK 
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "defines.hpp"
class unimplemented_error {};
class GenericLogicError : public std::logic_error {
 public:
    explicit GenericLogicError(const char* ch) : std::logic_error(ch) {
    }
    explicit GenericLogicError(std::string s) : std::logic_error(s) {
    }
};
class GenericRuntimeError : public std::runtime_error {
 public:
    explicit GenericRuntimeError(const char* ch) : std::runtime_error(ch) {
    }
    explicit GenericRuntimeError(std::string s) : std::runtime_error(s) {
    }
};
class WrongAnswerError : public std::runtime_error {
 public:
    WrongAnswerType type;
    explicit WrongAnswerError(WrongAnswerType type_, std::string msg_)
        : type(type_), std::runtime_error(msg_) {
    }
    explicit WrongAnswerError(WrongAnswerType type_)
        : WrongAnswerError(type_, "") {
    }
};
class OutputError : public std::runtime_error {
 public:
    explicit OutputError(const char* ch) : std::runtime_error(ch) {
    }
    explicit OutputError(std::string s) : std::runtime_error(s) {
    }
};
#define THROW_IF_DO_NOT_USE_(throwclass,cond,...) \
    do { \
        if (cond) { \
            std::stringstream ss; \
            ss << "[" << std::filesystem::path{__FILE__}.filename().c_str() \
               << ":" << __LINE__ << "] "; \
            throw throwclass(ss.str() + format_str(__VA_ARGS__)); \
        } \
    } while (0)
#define THROW_WA_IF_DO_NOT_USE_(type,cond,...) \
    do { \
        if (cond) { \
            std::stringstream ss; \
            ss << "[" << std::filesystem::path{__FILE__}.filename().c_str() \
               << ":" << __LINE__ << "] "; \
            throw WrongAnswerError(type, ss.str() + format_str(__VA_ARGS__)); \
        } \
    } while (0)
#define THROW_LOGIC_ERROR_IF(...) \
    THROW_IF_DO_NOT_USE_(GenericLogicError, __VA_ARGS__)
#define THROW_RUNTIME_ERROR_IF(...) \
    THROW_IF_DO_NOT_USE_(GenericRuntimeError, __VA_ARGS__)
#define THROW_OUTPUT_ERROR_IF(...) \
    THROW_IF_DO_NOT_USE_(OutputError, __VA_ARGS__)
#define THROW_WA_IF(type,...) THROW_WA_IF_DO_NOT_USE_(type, __VA_ARGS__)
#define WA_OR_RE_IF(type,...) \
    do { \
        if (type == WrongAnswerType::INVALID) { \
            THROW_RUNTIME_ERROR_IF(__VA_ARGS__); \
        } else { \
            THROW_WA_IF(type, __VA_ARGS__); \
        } \
    } while (0)
#define VALIDITY_CHECK_WITH_MSG(invalid_cond,v,m,...) \
    do { \
        if (invalid_cond) { \
            v = false; \
            if (m) { \
                *(m) += format_str(__VA_ARGS__); \
                *(m) += "\n"; \
            } \
        } \
    } while (0)
#endif
