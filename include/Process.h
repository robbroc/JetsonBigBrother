#ifndef PROCESS_H
#define PROCESS_H
#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/core.hpp>
#include "Person.h"
#include <chrono>
#include <string>
#include <utility>
#include "thread_utility.h"

typedef std::chrono::high_resolution_clock Clock;
struct bbox {
    int person_index = -1;
    cv::Rect rect;
    cv::Ptr<cv::Tracker> track;
    std::chrono::milliseconds timestamp; // aggiornato solo quando gli viene calcolata la rete di detection ci serve per scartare i bbox troppo vecchi
    std::chrono::milliseconds last_tracker_reinit;

    bbox(cv::Rect new_rect,cv::Mat frame): rect(new_rect),
    timestamp(std::chrono::duration_cast< std::chrono::milliseconds >
        (std::chrono::system_clock::now().time_since_epoch()))
    {
        track = cv::TrackerMOSSE::create();
        track->init(frame,rect);
        last_tracker_reinit = timestamp;
    }

    void update_time()
    {
        timestamp= std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());
    }

    void update_tracker(cv::Mat frame)
    {
        track.release();
        track = cv::TrackerMOSSE::create();
        //std::cout<<rect<<std::endl;
        track->init(frame,rect);
        last_tracker_reinit = std::chrono::duration_cast< std::chrono::milliseconds >
                              (std::chrono::system_clock::now().time_since_epoch());
    }
};

class Process
{
    public:
        Process(const std::string& path_dir_detection_model,const std::string& path_dir_reid_model);
        virtual ~Process();
        void run();
        static float get_overlap_perc(cv::Rect first, cv::Rect second);
        void get_corresponding_bbox(const cv::Rect& to_assoc, int &index, float &perc,int thr);
        std::vector<float> predict_face(cv::Mat frame, cv::Rect to_crop);
        int get_person(const std::vector<float>& face_feat, float thr);
        static float cosine_similarity(std::vector<float> , std::vector<float>);
        void show_result();
        bool capture_frame();
        void tracking_bboxes();
        void face_detection();
        void face_identification();


    protected:

    private:
        cv::VideoCapture camera;
        cv::Mat frame;
        std::vector<Person> db;
        std::vector<bbox> bboxes;
        std::string path_dir_detection_model;
        std::string path_dir_reid_model;
        cv::dnn::DetectionModel detect_model;
        cv::dnn::Net reid_model;
        short frame_counter = 0;
        std::vector<float> confidence;
        std::vector<int> classes;
        std::vector<cv::Rect> prediction_rect;
        std::map<int,std::vector<std::pair<float,cv::Rect> > > rect_to_assign;
        std::chrono::milliseconds ts;


};

#endif // PROCESS_H
