#include <fstream>
#include "world.hpp"
int main(int argc, char** argv) {
    if (argc == 1) {
        std::cerr << "Usage: ./playground <world_info_file> "
                     "<OPTIONAL:json_log_output_file>"
                  << std::endl;
        exit(1);
    }
    std::ifstream ifs(argv[1]);
    if (!ifs) {
        throw std::runtime_error("Failed to open the file:" +
                                 std::string(argv[1]));
    }
    std::ofstream logofs;
    World world;
    if (argc == 3) {
        logofs.open(argv[2]);
        world.set_json_log_output_stream(&logofs);
    }
    world.initialize();
    world.read_from_stream(ifs);
    world.interact(std::cin, std::cout);
}
