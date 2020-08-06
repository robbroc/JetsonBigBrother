#include "Process.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace std::chrono_literals;
#include <unistd.h> //da togliere
Process::Process(const string& path_dir_detection_model,const string& path_dir_reid_model):
 camera(0, cv::CAP_V4L2), // per la webcam
//camera("./out.mp4"), //per il debug DA TOGLIERE
detect_model(path_dir_detection_model+"optimized.pb",path_dir_detection_model+"cvgraph.pbtxt")
{
    camera.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G')); // facciamo una compressione per ridurre le tempistiche, ciò viene effettuato dalla camera
    detect_model.setInputParams(1.0,Size(320,320),0,true);
    detect_model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    detect_model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    reid_model=readNetFromCaffe(path_dir_reid_model+"model-caffe.prototxt", path_dir_reid_model+"model-caffe.caffemodel");

    reid_model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    reid_model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    cv::namedWindow("Frame", cv::WINDOW_NORMAL);
}

Process::~Process()
{
    //dtor
}
//calcola la predizione
std::vector<float> Process::predict_face(Mat frame, Rect to_crop)
{
    auto blob = cv::dnn::blobFromImage(frame(to_crop),1.0f,Size(112,112),Scalar(),true);

    static auto in1 = blob.clone(), in2 = blob.clone();

    //imshow("faccia", frame(to_crop));

    in1 = 127.5f;
    in2 = 0.0078125f;

    reid_model.setInput(in1, "minusscalar0_second");
    reid_model.setInput(in2, "mulscalar0_second");
    reid_model.setInput(blob, "data");

    auto res = reid_model.forward("pre_fc1");

    return res;
}
// calcola la percentuale di overlap tra due rect
float Process::get_overlap_perc(cv::Rect first, cv::Rect second)
{
    const auto t_0 = first.tl().y;
    const auto b_0 = first.br().y;
    const auto r_0 = first.br().x;
    const auto l_0 = first.tl().x;
    const auto t_1 = second.tl().y;
    const auto b_1 = second.br().y;
    const auto r_1 = second.br().x;
    const auto l_1 = second.tl().x;

    const auto a_overlap = (max(l_0,l_1) - min(r_0, r_1) ) *
                           (max(t_0,t_1) - min(b_0, b_1) );

    return a_overlap == 0 ? 0.0f : static_cast<float>(a_overlap)/static_cast<float>(first.area()) * 100.0f;
}
//dato il rettangolo da associare al bbox restituisce l'indice del bbox con cui overlappa maggiormente
// se non lo trova restituisce -1
void Process::get_corresponding_bbox(const cv::Rect& to_assoc, int &index, float &perc, int thr)
{
    perc = 0.0f;
    index = -1;
    for(size_t i = 0; i<bboxes.size();i++)
    { // bisogna distinguere in qualche modo i rect appena aggiunti da quelli che vanno poi identificati in un secondo tempo tramite feature extraction
        if(bboxes[i].person_index < -1) { // serve per fare in modo che la prossima volta che viene fatto il face identification,
            // quelli che hanno -1 su persona sono in realtà quelli inseriti nella precedente face identification
            // e non vengono esclusi dal calcolo della percentuale di overlap
            bboxes[i].person_index ++;
            continue; // per evitare di associare rect a bbox appenaaggiunti
        }

        auto actual_perc = get_overlap_perc(to_assoc,bboxes[i].rect);
        //cout<<actual_perc<<endl;
        if(static_cast<int>(actual_perc) > thr)
        {
            if(actual_perc > perc)
            {
                perc = actual_perc;
                index = i;
            }
        }
    }
}
//calcolo della cosine distance
float Process::cosine_similarity(std::vector<float> A, std::vector<float>B)
{
    float mul = 0.0;
    float d_a = 0.0;
    float d_b = 0.0;
    if (A.size() != B.size())
    {
        throw std::logic_error("Vector A and Vector B are not the same size");
    }

    // Prevent Division by zero
    if (A.size() < 1)
    {
        throw std::logic_error("Vector A and Vector B are empty");
    }
    auto B_iter = B.begin();
    auto A_iter = A.begin();
    for( ; A_iter != A.end(); A_iter++ , B_iter++ )
    {
        mul += *A_iter * *B_iter;
        d_a += *A_iter * *A_iter;
        d_b += *B_iter * *B_iter;
    }
    if (d_a == 0.0f || d_b == 0.0f)
    {
        throw std::logic_error(
                "cosine similarity is not defined whenever one or both "
                "input vectors are zero-vectors.");
    }
    return 1.0f-(mul / (sqrt(d_a) * sqrt(d_b)));
}
// restituisce il db[i]puntatore ad una persona,eventualmente aggiunge la persona nel db
int Process::get_person(const std::vector<float>& face_feat, float thr)
{
    int min_index = -1;
    float min_coss = thr;
    for(size_t i = 0; i < db.size(); i++)
    {
        float coss = cosine_similarity(face_feat, db[i].Getfeatures());
        if(coss<min_coss)
        {
            min_index = i;
            min_coss = coss;
        }
        //cout<<"similarity cos "<<coss<<endl;

    }
    //cout<<endl<<"--------------------------------------------------------"<<endl;

    if(min_index == -1) // creo la persona
    {
        //cout<<"aggiungo una persona al db con id ";
        db.emplace_back(face_feat);
        //cout<<db.back().Getid()<<endl; // DA TOGLIERE
        //std::cout << &db.back() << endl;
        return db.size()-1;
    }

    return min_index;
}


 // PIAZZARE L'ESECUZIONE DI QUESTA FUNZIONE SU UN ALTRO THREAD
void Process::show_result()
{
    unique_lock<mutex> lock1(frame_mut,defer_lock);
    unique_lock<mutex> lock2(bb_mut,defer_lock);
    std::lock(lock1, lock2);
    auto frame_to_show = frame.clone();
    for(auto && bb:bboxes)
    {
        cv::rectangle(frame_to_show,bb.rect, cv::Scalar(0,255,0));
        cv::putText(frame_to_show," id: "+to_string(bb.person_index),bb.rect.tl(),cv::FONT_HERSHEY_COMPLEX,0.4f,255);
    }
    //  mostra l'immagine
    cv::imshow("Frame",frame_to_show);
    if (waitKey(1) >= 0) // DA TOGLIERE
        exit(0);
    //cv::resizeWindow("Frame", 640, 480);
    //cv::waitKey(1); DA RIPRISTINARE

}

bool Process::capture_frame() {
    // per gestire l'accesso concorrente
    unique_lock<mutex> lock(frame_mut);
    ts = chrono::duration_cast<chrono::milliseconds >(chrono::system_clock::now().time_since_epoch());
    camera>> frame;
    if((++frame_counter) ==4) frame_counter=1;
    // per debug, controllo che il video sia finito ed esco DA TOGLIERE
    if (frame.empty()) {
        //cerr << "ERROR! blank frame grabbed\n";
        return false;
    }
    return true;

}

void Process::tracking_bboxes() {
    while(frame_counter==1){// deve controllare e cedere la cpu eventualmente ad ogni periodo
        sched_yield();
    }
    unique_lock<mutex> lock1(frame_mut,defer_lock);
    unique_lock<mutex> lock2(bb_mut,defer_lock);
    std::lock(lock1, lock2);
    for(auto i = 0; i<bboxes.size();i++)
    {
        if((ts - bboxes[i].timestamp) > 1000ms) // controllo quando è stato associato l'ultima volta un rect rilevato dalla rete di detection a questo bbox
        {   // se è passato più di un secondo lo scarto e continuo
            bboxes.erase(bboxes.begin()+i);
            i--;
            continue;
        }
        // se non è così vecchio effetuiamo il tracking
        Rect2d new_rect;

        bool tracked = bboxes[i].track->update(frame,new_rect);

        // controlliamo se il tracking è andato a buon fine eventualmente distruggiamo il bbox
        if(!tracked)
        {
            //cout<<"cancello bbox "<<i<<endl;
            bboxes.erase(bboxes.begin()+i);
            i--;
            continue;
        }
        bboxes[i].rect = new_rect;
    }
}

void Process::face_detection() {
    // controllo se è arrivato il turno di eseguire, sennò salto l'esecuzione
    while(frame_counter!=1){// deve controllare e cedere la cpu eventualmente ad ogni periodo
        sched_yield();
    }
    unique_lock<mutex> lock1(frame_mut,defer_lock);
    unique_lock<mutex> lock2(bb_mut,defer_lock);
    std::lock(lock1, lock2);
    detect_model.detect(frame,classes,confidence,prediction_rect,0.6f,0.001f);



    for(auto i = 0; i<prediction_rect.size();i++) // associamo le predizioni dei rettangoli ad un bbox esistente dove possibile, sennò creiamo un nuovo bbox
    {
        int index = -1;
        float perc = 0.0f;
        get_corresponding_bbox(prediction_rect[i],index,perc,20);

        if(index == -1)
        { // non è stata trovata nessuna corrispondenza
            // aggiungiamo il rect ad un bbox e quindi al vettore di bboxes
            bboxes.emplace_back(prediction_rect[i],frame,prediction_rect.size() - i);
        }
        else // aggiungiamo il rect alla mappa tra indici dei bbox e rect e perc
        {
            if(rect_to_assign.find(index) == rect_to_assign.end())
            { // necessario per creare il vettore la prima volta prima di fare il push_back
                rect_to_assign[index] = vector<pair<float,Rect> >();
                rect_to_assign[index].push_back(make_pair(perc,prediction_rect[i]));
            }
            else
                rect_to_assign[index].push_back(make_pair(perc,prediction_rect[i]));
        }
    }
    // ora bisogna decidere come assegnare

    for(auto && p:rect_to_assign)
    {//assegniamo al bbox il rect che interseca maggiormente
        float max_perc = 0.0f;
        Rect max_rect;
        for(auto && pair2:p.second)  //itero sul vettore di pair
        {   // cerco la percentuale maggiore
            if (pair2.first > max_perc)
            {
                max_perc = pair2.first;
                max_rect = pair2.second;

            }

        }

        bboxes[p.first].update_time();

        if((ts - bboxes[p.first].last_tracker_reinit) > 1000ms) {
            //if(max_perc > 0.1f) {
            // aggiorno il boundingbox e il suo timestamp
            //cout<<"aggiorno tutto il bbox "<<p.first<<endl;
            bboxes[p.first].rect=max_rect;
            bboxes[p.first].update_tracker(frame);
            }
        //}
    }
    // ripuliamo rect_to_assign
    rect_to_assign.clear();

}

void Process::face_identification() {
    while(frame_counter!=1){// deve controllare e cedere la cpu eventualmente ad ogni periodo
        sched_yield();
    }
    unique_lock<mutex> lock1(frame_mut,defer_lock);
    unique_lock<mutex> lock2(bb_mut,defer_lock);
    std::lock(lock1, lock2);
    // controlliamo se vanno effettuate delle face identification
    for(size_t i = 0; i<bboxes.size();i++)
    {
        if(bboxes[i].person_index != -1) continue; // se non è nullpointer salta
        // se è nullpointer va calcolata la rete
        // eventualmente questa parte la si parallelizza
        // calcolo le feature della faccia
        auto face_feat = predict_face(frame,bboxes[i].rect);


        // la cerco nel db
        bboxes[i].person_index = get_person(face_feat, 0.6f);


    }
}
