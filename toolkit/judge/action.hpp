#ifndef HEADER_2HC2022_ACTION
#define HEADER_2HC2022_ACTION 
#include <memory>
#include <string>
#include "defines.hpp"
#include "json_fwd.hpp"
class Worker;
namespace action {
#define ACTION_TYPES_ \
    def(INVALID) def(UNSPECIFIED) def(STAY) def(MOVE) def(EXECUTE)
enum class ActionType : int {
#define def(x) x,
    ACTION_TYPES_
#undef def
};
inline std::string action_type_to_str(ActionType type) {
#define def(x) #x,
    static constexpr const char* names[] = {ACTION_TYPES_};
#undef def
    return names[static_cast<std::underlying_type_t<ActionType>>(type)];
}
struct Action {
    discrete_time_t begin_time = INVALID_TIME;
    discrete_time_t target_end_time = INVALID_TIME;
    virtual void dispatch(discrete_time_t current_time, Worker* w) const = 0;
    static std::shared_ptr<action::Action> interpret(std::string action_str);
    virtual ActionType type() const {
        return ActionType::UNSPECIFIED;
    }
    virtual void to_json_impl(json_ref j) const = 0;
    friend void to_json(json_ref j, const Action& a);
    friend std::ostream& operator<<(std::ostream& os, const Action& a);
};
#define DEFINE_ACTION_REGEX(...) \
    inline static std::string action_regex() { \
        static std::string r = std::string(action_name) + __VA_ARGS__ + \
                               CONTESTANT_VALUE_DELIMITER + "*"; \
        return r; \
    } \
    struct __xd
struct Stay : public Action {
    static constexpr const char* action_name = "stay";
    DEFINE_ACTION_REGEX("");
    void dispatch(discrete_time_t current_time, Worker* w) const override;
    Stay() = default;
    explicit Stay(std::string s);
    ActionType type() const override {
        return ActionType::STAY;
    }
    void to_json_impl(json_ref j) const override;
};
struct Move : public Action {
    static constexpr const char* action_name = "move";
    DEFINE_ACTION_REGEX(R"( [1-9][0-9]*)");
    vertex_index_t to = INVALID_VERTEX_ID;
    void dispatch(discrete_time_t current_time, Worker* w) const override;
    Move() = default;
    explicit Move(std::string s);
    ActionType type() const override {
        return ActionType::MOVE;
    }
    void to_json_impl(json_ref j) const override;
};
struct Execute : public Action {
    static constexpr const char* action_name = "execute";
    DEFINE_ACTION_REGEX(R"( [1-9][0-9]* [1-9][0-9]*)");
    job_id_t job_id = INVALID_JOB_ID;
    int amount = INVALID_COUNT;
    void dispatch(discrete_time_t current_time, Worker* w) const override;
    Execute() = default;
    explicit Execute(std::string s);
    ActionType type() const override {
        return ActionType::EXECUTE;
    }
    void to_json_impl(json_ref j) const override;
};
}
#undef DEFINE_ACTION_REGEX
#endif
