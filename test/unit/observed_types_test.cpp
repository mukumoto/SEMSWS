// observed_types_test — ParseObservedType + channel-name round-trip.

#include "srcrecv/ObservedTypes.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

#define REQUIRE(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        std::exit(1); \
    } \
} while (0)

int main() {
    using namespace SEM;

    REQUIRE(ParseObservedType("PS")   == ReceiverType::Pressure);
    REQUIRE(ParseObservedType("VEL")  == ReceiverType::Velocity);
    REQUIRE(ParseObservedType("DISP") == ReceiverType::Displacement);
    REQUIRE(ParseObservedType("ACC")  == ReceiverType::Acceleration);

    REQUIRE(std::string(ObservedChannelName(ReceiverType::Pressure))     == "PS");
    REQUIRE(std::string(ObservedChannelName(ReceiverType::Velocity))     == "VEL");
    REQUIRE(std::string(ObservedChannelName(ReceiverType::Displacement)) == "DISP");
    REQUIRE(std::string(ObservedChannelName(ReceiverType::Acceleration)) == "ACC");

    REQUIRE(IsScalarObservedType(ReceiverType::Pressure));
    REQUIRE(!IsScalarObservedType(ReceiverType::Velocity));
    REQUIRE(IsVectorObservedType(ReceiverType::Velocity));
    REQUIRE(IsVectorObservedType(ReceiverType::Displacement));
    REQUIRE(IsVectorObservedType(ReceiverType::Acceleration));
    REQUIRE(!IsVectorObservedType(ReceiverType::Pressure));

    // Round-trip name -> type -> name.
    for (const char* s : {"PS", "VEL", "DISP", "ACC"}) {
        auto t = ParseObservedType(s);
        REQUIRE(std::string(ObservedChannelName(t)) == s);
    }
    return 0;
}
