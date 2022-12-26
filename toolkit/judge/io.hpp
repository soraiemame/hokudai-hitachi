#ifndef HEADER_2HC2022_IO
#define HEADER_2HC2022_IO 
// #include <filesystem>
#include <experimental/filesystem>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include "defines.hpp"
#include "error_check.hpp"
enum class ValueGroup {
    SCORE,
    COUNT,
    INDEX,
    PROBABILITY,
    COEFFICIENT,
    TIME,
    TYPEID,
    DISTANCE,
    STRING,
    WEATHER_VALUE,
    FLAG,
    NON_NEGATIVE_COUNT,
    POSITIVE_COUNT,
    CONTAINER,
    POSITIVE_DURATION,
    SEED,
    NONE,
    COORD,
};
class ValueReader {
    std::stringstream ssptr_;
 public:
    std::istream& is;
    explicit ValueReader(std::istream& is_) : is(is_) {
    }
    explicit ValueReader(std::stringstream&& ss)
        : ssptr_(std::move(ss)), is(ssptr_) {
    }
    bool operator!() const {
        return !is;
    }
};
class ValueWriter {
 public:
    std::ostream& os;
    explicit ValueWriter(std::ostream& os_) : os(os_) {
    }
};
template <class T> struct TaggedValue {
    T& ref_;
    ValueGroup grp;
    std::string comment;
    WrongAnswerType wa_type = WrongAnswerType::INVALID;
    TaggedValue(ValueGroup g_, T& r_, std::string comment_,
                WrongAnswerType wa_type_ = WrongAnswerType::INVALID)
        : ref_(r_), grp(g_), comment(comment_), wa_type(wa_type_) {
    }
    friend ValueReader& operator>>(ValueReader& reader, const TaggedValue& vr) {
        if constexpr (std::is_const_v<T>) {
            THROW_LOGIC_ERROR_IF(true,
                                 "Do not read data into const TaggedValue:%s",
                                 vr.comment.c_str());
        } else {
            switch (vr.grp) {
            case ValueGroup::NONE: {
                break;
            }
            case ValueGroup::DISTANCE:
            case ValueGroup::COUNT:
            case ValueGroup::SCORE: {
                reader.is >> vr.ref_;
                break;
            }
            case ValueGroup::SEED: {
                std::string s;
                reader.is >> s;
                WA_OR_RE_IF(
                    vr.wa_type,
                    !std::regex_match(s, std::regex(R"(0|[1-9]\d*)")),
                    "Invalid string for ValueGroup SEED (%s) (Regex:%s) :%s",
                    s.c_str(), R"(0|[1-9]\d*)", vr.comment.c_str());
                std::stringstream ss;
                ss << s;
                ss >> vr.ref_;
                break;
            }
            case ValueGroup::NON_NEGATIVE_COUNT: {
                reader.is >> vr.ref_;
                if constexpr (std::is_arithmetic_v<T> ||
                              std::is_same_v<remove_all_t<T>, float_score_t>) {
                    WA_OR_RE_IF(
                        vr.wa_type, vr.ref_ < 0,
                        "The specified value group is NON_NEGATIVE_COUNT "
                        "but the actual value is negative.(%d):%s",
                        (int)vr.ref_, vr.comment.c_str());
                }
                break;
            }
            case ValueGroup::POSITIVE_DURATION:
            case ValueGroup::POSITIVE_COUNT: {
                reader.is >> vr.ref_;
                if constexpr (std::is_arithmetic_v<T> ||
                              std::is_same_v<remove_all_t<T>, float_score_t>) {
                    std::stringstream ss;
                    ss << vr.ref_;
                    WA_OR_RE_IF(
                        vr.wa_type, vr.ref_ <= 0,
                        "The specified value group is "
                        "POSITIVE_COUNT/POSITIVE_DURATION "
                        "but the actual value is zero or negative.(%s):%s",
                        ss.str().c_str(), vr.comment.c_str());
                }
                break;
            }
            case ValueGroup::COEFFICIENT:
            case ValueGroup::COORD:
            case ValueGroup::PROBABILITY: {
                reader.is >> vr.ref_;
                break;
            }
            case ValueGroup::STRING: {
                reader.is >> vr.ref_;
                break;
            }
            case ValueGroup::TYPEID:
            case ValueGroup::TIME:
            case ValueGroup::WEATHER_VALUE:
            case ValueGroup::INDEX: {
                if constexpr (std::is_arithmetic_v<T> ||
                              std::is_same_v<remove_all_t<T>, float_score_t>) {
                    reader.is >> vr.ref_;
                    vr.ref_ = to_0b(vr.ref_);
                } else {
                    THROW_LOGIC_ERROR_IF(
                        true,
                        "Non-arithmetic values cannot be used for "
                        "TYPEID/TIME/WEATHER_VALUE/INDEX:%s",
                        vr.comment.c_str());
                }
                break;
            }
            case ValueGroup::FLAG: {
                if constexpr (std::is_same_v<bool, remove_all_t<T>>) {
                    int f = -1;
                    reader.is >> f;
                    if (f == 0) {
                        vr.ref_ = false;
                    } else if (f == 1) {
                        vr.ref_ = true;
                    } else {
                        WA_OR_RE_IF(vr.wa_type, true,
                                    "The value group specified is FLAG "
                                    "but the actual value is "
                                    "neither 0 nor 1.(%d):%s",
                                    f, vr.comment.c_str());
                    }
                } else {
                    THROW_LOGIC_ERROR_IF(true, "FLAG values must be boolean:%s",
                                         vr.comment.c_str());
                }
                break;
            }
            default:
                THROW_LOGIC_ERROR_IF(true, "Unsupported value group (in):%s",
                                     vr.comment.c_str());
            }
            WA_OR_RE_IF(vr.wa_type, !reader,
                        "Failed to read a tagged value.:%s",
                        vr.comment.c_str());
            return reader;
        }
    }
    friend ValueWriter& operator<<(ValueWriter& writer, const TaggedValue& vr) {
        switch (vr.grp) {
        case ValueGroup::NONE: {
            break;
        }
        case ValueGroup::DISTANCE:
        case ValueGroup::COUNT:
        case ValueGroup::SCORE: {
            writer.os << vr.ref_;
            break;
        }
        case ValueGroup::SEED: {
            if constexpr (std::is_arithmetic_v<T> ||
                          std::is_same_v<remove_all_t<T>, float_score_t>) {
                THROW_RUNTIME_ERROR_IF(
                    vr.ref_ < 0,
                    "The specified value group is SEED "
                    "but the actual value is negative.(%d):%s",
                    (int)vr.ref_, vr.comment.c_str());
            }
            writer.os << vr.ref_;
            break;
        }
        case ValueGroup::NON_NEGATIVE_COUNT: {
            if constexpr (std::is_arithmetic_v<T> ||
                          std::is_same_v<remove_all_t<T>, float_score_t>) {
                THROW_RUNTIME_ERROR_IF(
                    vr.ref_ < 0,
                    "The specified value group is NON_NEGATIVE_COUNT "
                    "but the actual value is negative.(%d):%s",
                    (int)vr.ref_, vr.comment.c_str());
            }
            writer.os << vr.ref_;
            break;
        }
        case ValueGroup::POSITIVE_DURATION:
        case ValueGroup::POSITIVE_COUNT: {
            if constexpr (std::is_arithmetic_v<T> ||
                          std::is_same_v<remove_all_t<T>, float_score_t>) {
                std::stringstream ss;
                ss << vr.ref_;
                THROW_RUNTIME_ERROR_IF(
                    vr.ref_ <= 0,
                    "The specified value group is "
                    "POSITIVE_COUNT/POSITIVE_DURATION "
                    "but the actual value is zero or negative.(%s):%s",
                    ss.str().c_str(), vr.comment.c_str());
            }
            writer.os << vr.ref_;
            break;
        }
        case ValueGroup::COEFFICIENT:
        case ValueGroup::COORD:
        case ValueGroup::PROBABILITY: {
            writer.os << vr.ref_;
            break;
        }
        case ValueGroup::STRING: {
            writer.os << vr.ref_;
            break;
        }
        case ValueGroup::TYPEID:
        case ValueGroup::TIME:
        case ValueGroup::WEATHER_VALUE:
        case ValueGroup::INDEX: {
            if constexpr (std::is_arithmetic_v<T> ||
                          std::is_same_v<remove_all_t<T>, float_score_t>) {
                writer.os << to_1b(vr.ref_);
            } else {
                THROW_LOGIC_ERROR_IF(true,
                                     "Non-arithmetic values cannot be used for "
                                     "TYPEID/TIME/WEATHER_VALUE/INDEX:%s",
                                     vr.comment.c_str());
            }
            break;
        }
        case ValueGroup::FLAG: {
            if constexpr (std::is_same_v<bool, remove_all_t<T>>) {
                int f = -1;
                if (vr.ref_ == true) {
                    f = 1;
                } else if (vr.ref_ == false) {
                    f = 0;
                } else {
                    THROW_LOGIC_ERROR_IF(true,
                                         "The value group specified is FLAG "
                                         "but the actual value is "
                                         "neither true nor false.:%s",
                                         vr.comment.c_str());
                }
                writer.os << f;
            } else {
                THROW_LOGIC_ERROR_IF(true, "FLAG values must be boolean:%s",
                                     vr.comment.c_str());
            }
            break;
        }
        case ValueGroup::CONTAINER: {
            writer.os << vr.ref_;
            break;
        }
        default:
            THROW_LOGIC_ERROR_IF(true, "Unsupported value group (out):%s",
                                 vr.comment.c_str());
        }
        return writer;
    }
};
#define TAG(vg,...) TAGWA(WrongAnswerType::INVALID, vg, __VA_ARGS__)
#ifndef ONLINE_JUDGE
#define TAGWA(wa_type,vg,...) \
    (TAG_(vg, __VA_ARGS__, "\n" + BASENAME_LINE_STR_ + #__VA_ARGS__, wa_type))
#else
#define TAGWA(wa_type,vg,...) (TAG_(vg, __VA_ARGS__, "", wa_type))
#endif
template <class T>
TaggedValue<T> TAG_(ValueGroup vg, T& var, std::string comment,
                    WrongAnswerType wa_type) {
    return TaggedValue(vg, var, comment, wa_type);
}
template <class T>
TaggedValue<const T> TAG_(ValueGroup vg, const T& var, std::string comment,
                          WrongAnswerType wa_type) {
    return TaggedValue(vg, var, comment, wa_type);
}
#define VA_(...) #__VA_ARGS__
#ifndef ONLINE_JUDGE
#define get_single_line_stream(...) \
    (get_single_line_stream_(__VA_ARGS__, "\n\t%s %s", BASENAME_LINE_C_STR_, \
                             VA_(__VA_ARGS__)))
#else
#define get_single_line_stream(...) (get_single_line_stream_(__VA_ARGS__))
#endif
template <class... Args>
inline ValueReader get_single_line_stream_(const ValueReader& r,
                                           Args&&... args) {
    std::string line;
    std::getline(r.is, line);
#ifndef ONLINE_JUDGE
    THROW_RUNTIME_ERROR_IF(!r, "Failed to read a line:%s",
                           format_str(std::forward<Args>(args)...).c_str());
#else
    THROW_RUNTIME_ERROR_IF(!r, "");
#endif
    std::stringstream ss;
    ss << line;
    return ValueReader(std::move(ss));
}
inline void
throw_if_nontrivial_chars_left(ValueReader& r,
                               WrongAnswerType wat = WrongAnswerType::INVALID) {
    char c;
    while (r.is.rdbuf()->in_avail() != 0) {
        r.is >> c;
        if (!r)
            break;
        WA_OR_RE_IF(
            wat, std::string(&c, 1) != std::string(CONTESTANT_VALUE_DELIMITER),
            "Some nontrivial characters are still left");
    }
}
template <class... Args>
void read_general_DO_NOT_USE_(ValueReader& is, bool no_chars_left,
                              TaggedValue<Args>&&... args) {
    (is >> ... >> std::forward<TaggedValue<Args>>(args));
    if (!is) {
        throw std::runtime_error("Failed to read values into args");
    }
    if (no_chars_left) {
        throw_if_nontrivial_chars_left(is);
    }
}
template <class... Args>
void readline_general_DO_NOT_USE_(ValueReader& is, bool no_chars_left,
                                  TaggedValue<Args>&&... args) {
    auto ss = get_single_line_stream(is);
    read_general_DO_NOT_USE_(ss, no_chars_left,
                             std::forward<TaggedValue<Args>>(args)...);
}
template <class... Args>
void readline(ValueReader& is, TaggedValue<Args>&&... args) {
    readline_general_DO_NOT_USE_(is, false,
                                 std::forward<TaggedValue<Args>>(args)...);
}
template <class... Args>
void readline_exact(ValueReader& is, TaggedValue<Args>&&... args) {
    readline_general_DO_NOT_USE_(is, true,
                                 std::forward<TaggedValue<Args>>(args)...);
}
template <class... Args>
void read_vars(ValueReader& is, TaggedValue<Args>&&... args) {
    read_general_DO_NOT_USE_(is, false,
                             std::forward<TaggedValue<Args>>(args)...);
}
template <class T, class Fn>
void read_n_vars(ValueReader& is, int n, ValueGroup vg, const Fn& f) {
    for (int i = 0; i < n; i++) {
        T v;
        read_general_DO_NOT_USE_(is, false, TAG(vg, v));
        f(std::move(v));
    }
}
template <class T, class Fn>
void read_n_vars_exact(ValueReader& is, int n, ValueGroup vg, const Fn& f,
                       WrongAnswerType wa_type = WrongAnswerType::INVALID) {
    read_n_vars<T>(is, n, vg, f);
    throw_if_nontrivial_chars_left(is, wa_type);
}
template <class... Args>
void output_line_to_contestant_no_flush_DO_NOT_USE_(
    ValueWriter& w, std::string end_str, TaggedValue<Args>&&... args) {
    const char* sep = "";
    (
        [&] {
            if (args.grp == ValueGroup::NONE) {
            } else {
                w.os << sep;
                w << std::forward<TaggedValue<Args>>(args);
                sep = CONTESTANT_VALUE_DELIMITER;
            }
        }(),
        ...);
    w.os << end_str;
}
template <class... Args>
void output_line_to_contestant_no_flush(ValueWriter& w,
                                        TaggedValue<Args>&&... args) {
    output_line_to_contestant_no_flush_DO_NOT_USE_(
        w, "\n", std::forward<TaggedValue<Args>>(args)...);
}
template <class... Args>
void output_line_to_contestant(ValueWriter& w, TaggedValue<Args>&&... args) {
    output_line_to_contestant_no_flush(
        w, std::forward<TaggedValue<Args>>(args)...);
    w.os << std::flush;
}
template <class... Args>
void concat_for_contestant(ValueWriter& w, TaggedValue<Args>&&... args) {
    output_line_to_contestant_no_flush_DO_NOT_USE_(
        w, "", std::forward<TaggedValue<Args>>(args)...);
}
template <class Container, class Fn> struct StreamAdapter {
    const Container& ref_;
    const Fn& f_;
    bool no_size = false;
    StreamAdapter(const Container& r, const Fn& f, bool no_size_ = false)
        : ref_(r), f_(f), no_size(no_size_) {
    }
    friend std::ostream& operator<<(std::ostream& os,
                                    const StreamAdapter& rhs) {
        ValueWriter w(os);
        auto it = rhs.ref_.begin();
        if (rhs.no_size) {
            if (rhs.ref_.size() == 0) {
            } else {
                rhs.f_(w, *(it++));
            }
        } else {
            w << TAG(ValueGroup::NON_NEGATIVE_COUNT, rhs.ref_.size());
        }
        for (; it != rhs.ref_.end(); it++) {
            w.os << CONTESTANT_VALUE_DELIMITER;
            rhs.f_(w, *it);
        }
        return os;
    }
};
class debug_null_buffer_DO_NOT_USE_ : public std::streambuf {
 public:
    int overflow(int c) {
        return c;
    }
};
class null_stream : public std::ostream {
 public:
    null_stream() : std::ostream(&m_sb) {
    }
 private:
    debug_null_buffer_DO_NOT_USE_ m_sb;
};
class LogInfoSwitcher {
 protected:
    mutable bool log_mutable_only_ = false;
    mutable discrete_time_t switched_time_ = INVALID_TIME;
 public:
    virtual void log_mutable_info_only(discrete_time_t crt0b, bool e) const {
        log_mutable_only_ = e;
        switched_time_ = crt0b;
    }
};
#endif
