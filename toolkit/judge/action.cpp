#include "action.hpp"
#include <memory>
#include <regex>
#include <sstream>
#include "lib/json.hpp"
#include "worker.hpp"
namespace action {
void to_json(json_ref j, const Action& a) {
    a.to_json_impl(j);
}
std::ostream& operator<<(std::ostream& os, const Action& a) {
    json j = a;
    os << j;
    return os;
}
}
action::Stay::Stay(std::string s) : Stay() {
#ifndef ONLINE_JUDGE
    THROW_WA_IF(WrongAnswerType::INVALID_STAY_STRING,
                !std::regex_match(s, std::regex(action_regex())),
                "Regex mismatch for Stay:%s", action_regex().c_str());
#endif
}
void action::Stay::to_json_impl(json_ref j) const {
    j = json{{"type", action_type_to_str(type())}};
}
void action::Stay::dispatch(discrete_time_t current_time, Worker* w) const {
    w->do_action(current_time, *this);
}
action::Move::Move(std::string s) {
#ifndef ONLINE_JUDGE
    THROW_WA_IF(WrongAnswerType::INVALID_MOVE_STRING,
                !std::regex_match(s, std::regex(action_regex())),
                "Regex mismatch for Move:%s", action_regex().c_str());
#endif
    std::stringstream ss;
    std::string name;
    ss << s;
    ValueReader r(ss);
    r >> TAG(ValueGroup::STRING, name) >> TAG(ValueGroup::INDEX, to);
}
void action::Move::dispatch(discrete_time_t current_time, Worker* w) const {
    w->do_action(current_time, *this);
}
void action::Move::to_json_impl(json_ref j) const {
    j = json{{"type", action_type_to_str(type())}, {"to", to}};
}
action::Execute::Execute(std::string s) {
#ifndef ONLINE_JUDGE
    THROW_WA_IF(WrongAnswerType::INVALID_EXECUTE_STRING,
                !std::regex_match(s, std::regex(action_regex())),
                "Regex mismatch for Execute:%s", action_regex().c_str());
#endif
    std::stringstream ss;
    std::string name;
    ss << s;
    ValueReader r(ss);
    r >> TAG(ValueGroup::STRING, name) >> TAG(ValueGroup::INDEX, job_id) >>
        TAG(ValueGroup::POSITIVE_COUNT, amount);
}
void action::Execute::dispatch(discrete_time_t current_time, Worker* w) const {
    w->do_action(current_time, *this);
}
void action::Execute::to_json_impl(json_ref j) const {
    j = json{{"type", action_type_to_str(type())},
             {"job_id", job_id},
             {"amount", amount}};
}
std::shared_ptr<action::Action> action::Action::interpret(std::string str) {
    std::stringstream ss;
    ss << str;
    std::string action_name;
    ss >> action_name;
#define RETURN_ACTION(name) \
    do { \
        if (action_name == action::name::action_name) { \
            return std::make_shared<action::name>(str); \
        } \
    } while (0)
    RETURN_ACTION(Stay);
    RETURN_ACTION(Move);
    RETURN_ACTION(Execute);
    THROW_WA_IF(WrongAnswerType::INVALID_ACTION_NAME, true,
                "Unknown action name");
}
