#ifndef HEADER_2HC2022_LOGGER
#define HEADER_2HC2022_LOGGER 
#include <cstdio>
// #include <filesystem>
#include <experimental/filesystem>
#include <sstream>
#define __FILENAME__ (std::filesystem::path{__FILE__}.filename().c_str())
#define DEFAULT_DEBUG_LEVEL 4
#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL DEFAULT_DEBUG_LEVEL
#endif
#if DEBUG_LEVEL >= 1
#define ERROR(...) \
    do { \
        fprintf(stderr, "[%s:% 4d] ERROR  : ", __FILENAME__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        fflush(stderr); \
    } while (0)
#else
#define ERROR(fmt,...) 
#endif
#if DEBUG_LEVEL >= 2
#define WARNING(...) \
    do { \
        fprintf(stderr, "[%s:% 4d] WARNING : ", __FILENAME__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        fflush(stderr); \
    } while (0)
#else
#define WARNING(fmt,...) 
#endif
#if DEBUG_LEVEL >= 3
#define INFO(...) \
    do { \
        fprintf(stderr, "[%s:% 4d] INFO   : ", __FILENAME__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        fflush(stderr); \
    } while (0)
#else
#define INFO(fmt,...) 
#endif
#if DEBUG_LEVEL >= 4
#define DEBUG(...) \
    do { \
        fprintf(stderr, "[%s:% 4d] DEBUG  : ", __FILENAME__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        fflush(stderr); \
    } while (0)
#else
#define DEBUG(fmt,...) 
#endif
#if DEBUG_LEVEL >= 5
#define VERBOSE(...) \
    do { \
        fprintf(stderr, "[%s:% 4d] VERBOSE  : ", __FILENAME__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        fflush(stderr); \
    } while (0)
#else
#define VERBOSE(fmt,...) 
#endif
#if DEBUG_LEVEL >= 6
#define VERYVERBOSE(...) \
    do { \
        fprintf(stderr, "[%s:% 4d] VERBOSE  : ", __FILENAME__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        fflush(stderr); \
    } while (0)
#else
#define VERYVERBOSE(fmt,...) 
#endif
#if DEBUG_LEVEL >= 1
#define ERRORV(var) \
    do { \
        std::stringstream ss; \
        ss << var; \
        ERROR(#var " = %s", ss.str().c_str()); \
    } while (0)
#define WARNINGV(var) \
    do { \
        std::stringstream ss; \
        ss << var; \
        WARNING(#var " = %s", ss.str().c_str()); \
    } while (0)
#define INFOV(var) \
    do { \
        std::stringstream ss; \
        ss << var; \
        INFO(#var " = %s", ss.str().c_str()); \
    } while (0)
#define DEBUGV(var) \
    do { \
        std::stringstream ss; \
        ss << var; \
        DEBUG(#var " = %s", ss.str().c_str()); \
    } while (0)
#define VERBOSEV(var) \
    do { \
        std::stringstream ss; \
        ss << var; \
        VERBOSE(#var " = %s", ss.str().c_str()); \
    } while (0)
#define VERYVERBOSEV(var) \
    do { \
        std::stringstream ss; \
        ss << var; \
        VERBOSE(#var " = %s", ss.str().c_str()); \
    } while (0)
#else
#define ERRORV(var) 
#define WARNINGV(var) 
#define INFOV(var) 
#define DEBUGV(var) 
#define VERBOSEV(var) 
#define VERYVERBOSEV(var) 
#endif
#endif
