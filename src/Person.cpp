#include "Person.h"
unsigned int Person::id_counter = 0;

Person::Person(std::vector<float> feat):
id(id_counter++), features(feat)
{
    //ctor
}

Person::~Person()
{
    //dtor
}
