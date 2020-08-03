#ifndef PERSON_H
#define PERSON_H
#include <vector>

class Person
{
    public:
        Person(std::vector<float> feat);
        virtual ~Person();

        unsigned int Getid() { return id; }
        void Setid(unsigned int val) { id = val; }
        std::vector<float> Getfeatures() { return features; }
        void Setfeatures(std::vector<float> val) { features = val; }

    protected:

    private:
        static unsigned int id_counter;
        unsigned int id;
        std::vector<float> features;
};

#endif // PERSON_H
