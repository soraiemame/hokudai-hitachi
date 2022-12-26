#ifndef HEADER_2HC2022_FIFO
#define HEADER_2HC2022_FIFO 
#include <sys/stat.h>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
class TemporaryFIFO {
    std::filesystem::path tmpdir_;
    std::filesystem::path fifo_in_;
    std::filesystem::path fifo_out_;
    bool opened_ = false;
 public:
    static constexpr const char* IN_NAME = "in";
    static constexpr const char* OUT_NAME = "out";
    static constexpr const char DIR_TEMPLATE_NAME[] = "fifo-XXXXXX";
    std::filesystem::path in() const {
        if (!opened_) {
            throw std::logic_error("fifo_in is not opened yet.");
        }
        return fifo_in_;
    }
    std::filesystem::path out() const {
        if (!opened_) {
            throw std::logic_error("fifo_out is not opened yet.");
        }
        return fifo_out_;
    }
    void open() {
        if (opened_) {
            throw std::logic_error("FIFOs are already opened.");
        }
        {
            char dir_template[sizeof(DIR_TEMPLATE_NAME)] = {};
            std::memcpy(&dir_template[0], &DIR_TEMPLATE_NAME[0],
                        sizeof(DIR_TEMPLATE_NAME));
            const char* dir = mkdtemp(dir_template);
            if (dir == NULL) {
                throw std::runtime_error(
                    "Failed to create a temporary directory for FIFOs");
            }
            tmpdir_ = std::filesystem::path(dir);
        }
        {
            const std::filesystem::path in_ = tmpdir_ / IN_NAME;
            if (mkfifo(in_.c_str(), 0666) == -1) {
                std::filesystem::remove(tmpdir_);
                throw std::runtime_error("Failed to create fifo_in");
            }
            fifo_in_ = in_;
        }
        {
            const std::string out_ = tmpdir_ / OUT_NAME;
            if (mkfifo(out_.c_str(), 0666) == -1) {
                std::filesystem::remove(fifo_in_);
                std::filesystem::remove(tmpdir_);
                throw std::runtime_error("Failed to create fifo_out");
            }
            fifo_out_ = out_;
        }
        opened_ = true;
    }
    void close() {
        if (opened_) {
            std::filesystem::remove(out());
            std::filesystem::remove(in());
            std::filesystem::remove(tmpdir_);
            opened_ = false;
        }
    }
    ~TemporaryFIFO() {
        close();
    }
};
#endif
