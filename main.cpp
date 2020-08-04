#include "Process.h"
/*
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
// */
Process proc("./face_detection/","./face_identification/");
void cap_frame()
{
    while(true)
    {
        proc.capture_frame();
        sched_yield();
    }
}

void face_det()
{
    while(true)
    {
        proc.face_detection();
        sched_yield();
    }
}

void face_id()
{
    while(true)
    {
        proc.face_identification();
        sched_yield();
    }
}

void track_bb()
{
    while(true)
    {
        proc.tracking_bboxes();
        sched_yield();
    }
}

void show_im()
{
    while(true) {
        proc.show_result();
        sched_yield();
    }
}

int main()
{

    // tutto ciò risulta necessario per poter usare la GPU con sched_deadline
// e per poter gestire il thread che dialoga con la gpu ad alta priorità, evitando così che funga da collo di bottiglia
    sched_param fifo_params;
    fifo_params.sched_priority = 99;
    sched_setscheduler(pthread_self(), SCHED_FIFO,&fifo_params);
    /*int CudaDevice = cv::cuda::getDevice();
    cv::cuda::setDevice(CudaDevice); */
    data_2_pass param_cap_f, param_face_det, param_face_id, param_track_bb, param_show_im;
    // parametri del capture frame 25 fps
    param_cap_f.fun = &cap_frame;
    param_cap_f.period = 60 * 1000 * 1000;
    param_cap_f.deadline = 4 * 1000 * 1000;
    param_cap_f.runtime = 4 * 1000 * 1000;
    // parametri del face detection
    param_face_det.fun = &face_det;
    param_face_det.period = 60 * 1000 * 1000;
    param_face_det.deadline = 58 * 1000 * 1000;
    param_face_det.runtime =  53 * 1000 * 1000;
    // parametri del cface id
    param_face_id.fun = &face_id;
    param_face_id.period = 180 * 1000 * 1000;
    param_face_id.deadline = 180 * 1000 * 1000;
    param_face_id.runtime = 80 * 1000 * 1000;
    // parametri del track bb
    param_track_bb.fun = &track_bb;
    param_track_bb.period = 60 * 1000 * 1000;
    param_track_bb.deadline = 59 * 1000 * 1000;
    param_track_bb.runtime = 5 * 1000 * 1000;
    // parametri del show im
    param_show_im.fun = &show_im;
    param_show_im.period = 60 * 1000 * 1000;
    param_show_im.deadline = 60 * 1000 * 1000;
    param_show_im.runtime = 2 * 1000 * 1000;

    pthread_t thread_cap_f, thread_face_det, thread_face_id, thread_track_bb, thread_show_im;

	pthread_create(&thread_cap_f, nullptr, run_deadline , (void*)(&param_cap_f));
    pthread_create(&thread_face_det, nullptr, run_deadline , (void*)(&param_face_det));
    pthread_create(&thread_face_id, nullptr, run_deadline , (void*)(&param_face_id));
    pthread_create(&thread_track_bb, nullptr, run_deadline , (void*)(&param_track_bb));
    pthread_create(&thread_show_im, nullptr, run_deadline , (void*)(&param_show_im));
    // join di tutti i thread
	pthread_join(thread_cap_f, nullptr);
    pthread_join(thread_face_det, nullptr);
    pthread_join(thread_face_id, nullptr);
    pthread_join(thread_track_bb, nullptr);
    pthread_join(thread_show_im, nullptr);

}



