/* 
 * Copyright (C) 2011 Department of Robotics Brain and Cognitive Sciences - Istituto Italiano di Tecnologia
 * Author: Ugo Pattacini
 * email:  ugo.pattacini@iit.it
 * Permission is granted to copy, distribute, and/or modify this program
 * under the terms of the GNU General Public License, version 2 or any
 * later version published by the Free Software Foundation.
 *
 * A copy of the license can be found at
 * http://www.robotcub.org/icub/license/gpl.txt
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details
*/

#include <sstream>
#include <limits>
#include <algorithm>
#include <set>

#include <yarp/math/Math.h>
#include <yarp/math/Rand.h>

#include "module.h"

using namespace yarp::math;


/**********************************************************/
class BusyGate
{
    bool &gate;
    bool  owner;
public:
    /**********************************************************/
    BusyGate(bool &g) : gate(g)
    {
        gate=true;
        owner=true;
    }

    /**********************************************************/
    void release()
    {
        owner=false;
    }

    /**********************************************************/
    ~BusyGate()
    {
        if (owner)
            gate=false; 
    }
};


/**********************************************************/
Tracker::Tracker(const string &trackerType_, const double trackerTmo_) :
                 trackerType(trackerType_), trackerState(idle),
                 trackerTmo(trackerTmo_), trackerTimer(0.0)
{
    trackerResult.x=trackerResult.y=0;
    trackerResult.width=trackerResult.height=0;
}


/**********************************************************/
void Tracker::prepare()
{
    if (trackerState==no_need)
        trackerState=init;
}


/**********************************************************/
void Tracker::latchBBox(const cv::Rect &bbox)
{
    trackerResult.x=bbox.x;
    trackerResult.y=bbox.y;
    trackerResult.width=bbox.width;
    trackerResult.height=bbox.height;
    trackerState=no_need;
}


/**********************************************************/
void Tracker::track(const Image &img)
{
    cv::Mat frame=cv::cvarrToMat(img.getIplImage());
    if (trackerState==init)
    {
        if (trackerType=="MIL")
            tracker=cv::TrackerMIL::createTracker();
        else if (trackerType=="TLD")
            tracker=cv::TrackerTLD::createTracker();
        else if (trackerType=="KCF")
            tracker=cv::TrackerKCF::createTracker();
        else
            tracker=cv::TrackerBoosting::createTracker();
        
        tracker->init(frame,trackerResult);
        trackerTimer=Time::now();
        trackerState=tracking;
    }
    else if (trackerState==tracking)
    {
        if (Time::now()-trackerTimer<trackerTmo)            
        {
            tracker->update(frame,trackerResult);
            cv::Point tl((int)trackerResult.x,(int)trackerResult.y);
            cv::Point br(tl.x+(int)trackerResult.width,tl.y+(int)trackerResult.height);
            if ((tl.x<5) || (br.x>frame.cols-5) ||
                (tl.y<5) || (br.y>frame.rows-5))
                trackerState=idle;
        }
        else
            trackerState=idle;
    }
}


/**********************************************************/
bool Tracker::is_tracking(cv::Rect &bbox) const
{
    bbox=cv::Rect((int)trackerResult.x,(int)trackerResult.y,
                  (int)trackerResult.width,(int)trackerResult.height);
    return (trackerState!=idle);
}


/**********************************************************/
int Manager::processHumanCmd(const Bottle &cmd, Bottle &b)
{
    int ret=Vocab::encode(cmd.get(0).asString().c_str());
    b.clear();

    if (cmd.size()>1)
    {
        if (cmd.get(1).isList())
            b=*cmd.get(1).asList();
        else
            b=cmd.tail();
    }

    return ret;
}


/**********************************************************/
Bottle Manager::skimBlobs(const Bottle &blobs)
{
    Bottle skimmedBlobs;
    for (int i=0; i<blobs.size(); i++)
    {
        cv::Point cog=getBlobCOG(blobs,i);
        if ((cog.x==RET_INVALID) || (cog.y==RET_INVALID))
            continue;

        // skim out blobs that are too far in the cartesian space
        Vector x;
        if (get3DPosition(cog,x))
        {
            if ((x[0]>skim_blobs_x_bounds[0])&&(x[0]<skim_blobs_x_bounds[1])&&
                (x[1]>skim_blobs_y_bounds[0])&&(x[1]<skim_blobs_y_bounds[1]))
                skimmedBlobs.add(blobs.get(i));
        }
    }

    return skimmedBlobs;
}


/**********************************************************/
bool Manager::thresBBox(cv::Rect &bbox, const Image &img)
{
    cv::Point tl(bbox.x,bbox.y);
    cv::Point br(tl.x+bbox.width,tl.y+bbox.height);
    tl.x=std::min(img.width(),std::max(tl.x,0));
    tl.y=std::min(img.height(),std::max(tl.y,0));
    br.x=std::min(img.width(),std::max(br.x,0));
    br.y=std::min(img.height(),std::max(br.y,0));

    bbox=cv::Rect(tl.x,tl.y,br.x-tl.x,br.y-tl.y);
    if ((bbox.width>tracker_min_blob_size[0]) &&
        (bbox.height>tracker_min_blob_size[1]))
        return true;
    else
        return false;
}


/**********************************************************/
Bottle Manager::getBlobs()
{
    // grab resources
    mutexResources.lock();

    if (Bottle *pBlobs=blobExtractor.read(false))
    {
        lastBlobsArrivalTime=Time::now();
        lastBlobs=skimBlobs(*pBlobs);
        yInfo("Received blobs list: %s",lastBlobs.toString().c_str());
        
        if (lastBlobs.size()==1)
        {
            if (lastBlobs.get(0).asVocab()==Vocab::encode("empty"))
                lastBlobs.clear();
        }
    }
    else if (Time::now()-lastBlobsArrivalTime>blobs_detection_timeout)
        lastBlobs.clear();

    // release resources
    mutexResources.unlock();
    
    return lastBlobs;
}


/**********************************************************/
cv::Point Manager::getBlobCOG(const Bottle &blobs, const int i)
{
    cv::Point cog(RET_INVALID,RET_INVALID);
    if ((i>=0) && (i<blobs.size()))
    {
        cv::Point tl,br;
        Bottle *item=blobs.get(i).asList();
        if (item==NULL)
            return cog;

        tl.x=(int)item->get(0).asDouble();
        tl.y=(int)item->get(1).asDouble();
        br.x=(int)item->get(2).asDouble();
        br.y=(int)item->get(3).asDouble();

        cog.x=(tl.x+br.x)>>1;
        cog.y=(tl.y+br.y)>>1;
    }

    return cog;
}


/**********************************************************/
bool Manager::get3DPosition(const cv::Point &point, Vector &x)
{
    x.resize(3,0.0);
    if (rpcGet3D.getOutputCount()>0)
    {
        //New command for getting 3D point from depth camera
        Bottle cmd, reply;
        cmd.addString("get_3D_points");
        Bottle &content=cmd.addList();
        Bottle &vector=content.addList();
        vector.addDouble(point.x);
        vector.addDouble(point.y);
        cmd.addInt(0);

        mutexGet3D.lock();
        yInfo("Sending get3D query: %s",cmd.toString().c_str());
        rpcGet3D.write(cmd,reply);
        yInfo("Received blob cartesian coordinates: %s",reply.toString().c_str());
        mutexGet3D.unlock();

        Bottle *points=reply.get(0).asList();
        if (points!=NULL)
        {
            Bottle *point=points->get(0).asList();
            if (point->size()>0)
            {
                Bottle *in=point->get(0).asList();

                if (in->size()>0)
                {
                    x[0]=in->get(0).asDouble();
                    x[1]=in->get(1).asDouble();
                    x[2]=in->get(2).asDouble();

                    fromCameraToRoot(x);
                }
                else
                    yError()<<"Wrong 3D point dimensions";
            }
            else
                yError()<<"No point received from depth camera";
        }
        else
            yError()<<"Problems in receiving reply from depth camera";
    }

    yInfo()<<"x=("<<x.toString(3,3)<<")";

    return (norm(x)>0.0);
}

/***********************************************************************/
void Manager::fromCameraToRoot(Vector &x_to_be_rotated)
{
    Property *frame_info=cerGazePort.read(false);
    Vector x(3);
    Vector o(4);

    if (frame_info!=NULL)
    {
        Bottle &pose_b=frame_info->findGroup("depth");
        Bottle *pose=pose_b.get(1).asList();
        x[0]=pose->get(0).asDouble();
        x[1]=pose->get(1).asDouble();
        x[2]=pose->get(2).asDouble();
        
        o[0]=pose->get(3).asDouble();
        o[1]=pose->get(4).asDouble();
        o[2]=pose->get(5).asDouble();
        o[3]=pose->get(6).asDouble();

        Matrix H;
        H.resize(4,4);
        H=axis2dcm(o);
        H.setSubcol(x,0,3);
        H(3,3)=1;
        
        if ((norm(x)>0.0) && (norm(o)>0.0))
        {           
            Vector aux;
            aux.resize(4,1.0);
            aux.setSubvector(0,x_to_be_rotated);

            x_to_be_rotated=(H*aux).subVector(0,2);        
        }
    }
    else
      yError()<<"Frame info null";
}


/**********************************************************/
void Manager::acquireImage(const bool rtlocalization)
{
    // grab resources
    mutexResources.lock();

    // wait for incoming image
    if (ImageOf<PixelBgr> *tmp=imgIn.read())
    {
        if (rtlocalization)
            imgRtLoc=*tmp;
        else
            img=*tmp;
    }
    
    // release resources
    mutexResources.unlock();
}


/**********************************************************/
void Manager::drawBlobs(const Bottle &blobs, const int i,
                        Bottle *scores)
{
    // grab resources
    mutexResources.lock();

    BufferedPort<ImageOf<PixelBgr> > *port=(scores==NULL)?&imgOut:&imgRtLocOut;
    if (port->getOutputCount()>0)
    {
        cv::Scalar highlight(0,255,0);
        cv::Scalar lowlight(150,125,125);            

        // latch image
        ImageOf<PixelBgr> img=(scores==NULL)?this->img:this->imgRtLoc;
        cv::Mat imgMat=cv::cvarrToMat(img.getIplImage());
        for (int j=0; j<blobs.size(); j++)
        {
            cv::Point tl,br,txtLoc;
            Bottle *item=blobs.get(j).asList();
            tl.x=(int)item->get(0).asDouble();
            tl.y=(int)item->get(1).asDouble();
            br.x=(int)item->get(2).asDouble();
            br.y=(int)item->get(3).asDouble();
            txtLoc.x=tl.x;
            txtLoc.y=tl.y-5;

            ostringstream tag;
            tag<<"blob_"<<j;

            if (scores!=NULL)
            {
                // find the blob name (or unknown)
                string object=db.findName(*scores,tag.str());
                tag.str("");
                tag.clear();
                tag<<object;
            }

            cv::rectangle(imgMat,tl,br,(j==i)?highlight:lowlight,2);
            cv::putText(imgMat,tag.str().c_str(),txtLoc,cv::FONT_HERSHEY_SIMPLEX,
                        0.5,(j==i)?highlight:lowlight,2);
        }

        port->prepare()=img;
        port->write();
    }

    // release resources
    mutexResources.unlock();
}


/**********************************************************/
void Manager::rotate(cv::Mat &src, const double angle,
                     cv::Mat &dst)
{
    int len=std::max(src.cols,src.rows);
    cv::Point2f pt(len/2.0f,len/2.0f);
    cv::Mat r=cv::getRotationMatrix2D(pt,angle,1.0);
    cv::warpAffine(src,dst,r,cv::Size(len,len));
}


/**********************************************************/
void Manager::drawScoresHistogram(const Bottle &blobs,
                                  const Bottle &scores, const int i)
{
    if (imgHistogram.getOutputCount()>0)
    {
        // grab resources
        mutexResources.lock();

        // create image containing histogram
        ImageOf<PixelBgr> imgConf;
        imgConf.resize(600,600); imgConf.zero();

        // opencv wrappers
        cv::Mat imgRtLocMat=cv::cvarrToMat(imgRtLoc.getIplImage());
        cv::Mat imgConfMat=cv::cvarrToMat(imgConf.getIplImage());

        ostringstream tag;
        tag<<"blob_"<<i;

        // process scores on the given blob
        if (Bottle *blobScores=scores.find(tag.str().c_str()).asList())
        {
            // set up some variables and constraints
            int maxHeight=(int)(imgConf.height()*0.8);
            int minHeight=imgConf.height()-20;
            int widthStep=(blobScores->size()>0)?(int)(imgConf.width()/blobScores->size()):0;
            set<string> gcFilters;

            // cycle over classes
            for (int j=0; j<blobScores->size(); j++)
            {
                Bottle *item=blobScores->get(j).asList();
                if (item==NULL)
                    continue;

                string name=item->get(0).asString().c_str();
                double score=std::max(std::min(item->get(1).asDouble(),1.0),0.0);

                // smooth out quickly varying scores
                map<string,Filter*>::iterator it=histFiltersPool.find(name);

                // create filter if not available
                if (it==histFiltersPool.end())
                {
                    Vector num(histFilterLength,1.0);
                    Vector den(histFilterLength,0.0); den[0]=histFilterLength;
                    histFiltersPool[name]=new Filter(num,den,Vector(1,score));
                }
                else
                {
                    Vector scoreFilt=it->second->filt(Vector(1,score));
                    score=scoreFilt[0];
                }

                // put the class name in a convenient set for garbage collection
                gcFilters.insert(name);

                int classHeight=std::min(minHeight,imgConf.height()-(int)(maxHeight*score));

                cv::rectangle(imgConfMat,cv::Point(j*widthStep,classHeight),cv::Point((j+1)*widthStep,minHeight),
                              histColorsCode[j%(int)histColorsCode.size()],CV_FILLED);

                cv::Mat textImg=cv::Mat::zeros(imgConf.height(),imgConf.width(),CV_8UC3);
                cv::putText(textImg,name.c_str(),cv::Point(imgConf.width()-580,(j+1)*widthStep-10),
                            cv::FONT_HERSHEY_SIMPLEX,0.8,cv::Scalar(255,255,255),2);
                rotate(textImg,90.0,textImg);

                cv::Mat orig=cv::cvarrToMat(imgConf.getIplImage());
                orig=orig+textImg;
            }

            // draw the blob snapshot
            cv::Point tl,br,sz;
            Bottle *item=blobs.get(i).asList();
            tl.x=(int)item->get(0).asDouble();
            tl.y=(int)item->get(1).asDouble();
            br.x=(int)item->get(2).asDouble();
            br.y=(int)item->get(3).asDouble();
            sz.x=br.x-tl.x;
            sz.y=br.y-tl.y;            

            // copy the blob
            ImageOf<PixelBgr> imgTmp1;
            imgTmp1.resize(sz.x,sz.y);
            cv::Mat imgRtLocRoi(imgRtLocMat,cv::Rect(tl.x,tl.y,sz.x,sz.y));
            cv::Mat imgTmp1Mat=cv::cvarrToMat(imgTmp1.getIplImage());
            imgRtLocRoi.copyTo(imgTmp1Mat);

            // resize the blob
            ImageOf<PixelBgr> imgTmp2;
            int magFact=2;  // magnifying factor
            imgTmp2.resize(magFact*imgTmp1.width(),magFact*imgTmp1.height());
            cv::Mat imgTmp2Mat=cv::cvarrToMat(imgTmp2.getIplImage());
            cv::resize(imgTmp1Mat,imgTmp2Mat,imgTmp2Mat.size());

            // superimpose the blob on the histogram
            cv::Mat imgConfRoi(imgConfMat,cv::Rect(0,0,imgTmp2.width(),imgTmp2.height()));
            imgTmp2Mat.copyTo(imgConfRoi);
            cv::rectangle(imgConfMat,cv::Point(0,0),cv::Point(imgTmp2.width(),imgTmp2.height()),
                          cv::Scalar(255,255,255),3);

            // give chance for disposing filters that are no longer used (one at time)
            if ((int)histFiltersPool.size()>blobScores->size())
            {
                for (map<string,Filter*>::iterator it=histFiltersPool.begin();
                     it!=histFiltersPool.end(); it++)
                {
                    if (gcFilters.find(it->first)==gcFilters.end())
                    {
                        delete it->second;
                        histFiltersPool.erase(it);
                        break;
                    }
                }
            }
        }

        imgHistogram.prepare()=imgConf;
        imgHistogram.write();

        // release resources
        mutexResources.unlock();
    }
}


/**********************************************************/
int Manager::findClosestBlob(const Bottle &blobs, const cv::Point &loc)
{
    int ret=RET_INVALID;
    double min_d2=std::numeric_limits<double>::max();

    for (int i=0; i<blobs.size(); i++)
    {
        cv::Point cog=getBlobCOG(blobs,i);
        if ((cog.x==RET_INVALID) || (cog.y==RET_INVALID))
            continue;

        double dx=loc.x-cog.x;
        double dy=loc.y-cog.y;
        double d2=dx*dx+dy*dy;

        if (d2<min_d2)
        {
            min_d2=d2;
            ret=i;
        }
    }

    return ret;
}


/**********************************************************/
int Manager::findClosestBlob(const Bottle &blobs, const Vector &loc)
{
    int ret=RET_INVALID;
    double curMinDist=std::numeric_limits<double>::max();

    for (int i=0; i<blobs.size(); i++)
    {
        cv::Point cog=getBlobCOG(blobs,i);
        if ((cog.x==RET_INVALID) || (cog.y==RET_INVALID))
            continue;

        Vector x;
        if (get3DPosition(cog,x))
        {
            double dist=norm(loc-x);
            if (dist<curMinDist)
            {
                ret=i;
                curMinDist=dist;
            }
        }
    }

    return ret;
}


/**********************************************************/
Bottle Manager::classify(const Bottle &blobs,
                         const bool rtlocalization)
{
    // grab resources
    mutexResources.lock();

    if (rtlocalization)
        imgClassifier.write(imgRtLoc);
    else
        imgClassifier.write(img);

    Bottle cmd,reply;
    cmd.addVocab(Vocab::encode("classify"));
    Bottle &options=cmd.addList();
    for (int i=0; i<blobs.size(); i++)
    {
        ostringstream tag;
        tag<<"blob_"<<i;
        Bottle &item=options.addList();
        item.addString(tag.str().c_str());
        item.addList()=*blobs.get(i).asList();
    }
    yInfo("Sending classification request: %s",cmd.toString().c_str());
    rpcClassifier.write(cmd,reply);
    yInfo("Received reply: %s",reply.toString().c_str());

    // release resources
    mutexResources.unlock();

    return reply;
}


/**********************************************************/
void Manager::burst(const string &tag)
{
    if (trainBurst && (tag!=""))
    {
        Bottle cmd,reply;
        cmd.addVocab(Vocab::encode("burst"));
        cmd.addVocab(Vocab::encode(tag.c_str()));

        yInfo("Sending burst training request: %s",cmd.toString().c_str());
        rpcClassifier.write(cmd,reply);
        yInfo("Received reply: %s",reply.toString().c_str());
    }
}


/**********************************************************/
void Manager::train(const string &object, const Bottle &blobs,
                    const int i)
{
    // grab resources
    mutexResources.lock();

    imgClassifier.write(img);

    Bottle cmd,reply;
    cmd.addVocab(Vocab::encode("train"));
    Bottle &options=cmd.addList().addList();
    options.addString(object.c_str());

    if (i<0)
    {
        Vector z=zeros(4);
        options.addList().read(z);
    }
    else
        options.add(blobs.get(i));

    yInfo("Sending training request: %s",cmd.toString().c_str());
    rpcClassifier.write(cmd,reply);
    yInfo("Received reply: %s",reply.toString().c_str());

    if (trainOnFlipped && (i>=0))
    {
        ImageOf<PixelBgr> imgFlipped=img;
        cv::Mat imgFlippedMat=cv::cvarrToMat(imgFlipped.getIplImage());

        if (Bottle *item=blobs.get(i).asList())
        {
            cv::Point tl,br;
            tl.x=(int)item->get(0).asDouble();
            tl.y=(int)item->get(1).asDouble();
            br.x=(int)item->get(2).asDouble();
            br.y=(int)item->get(3).asDouble();

            cv::Mat roi(imgFlippedMat,cv::Rect(tl.x,tl.y,br.x-tl.x,br.y-tl.y));
            cv::flip(roi,roi,1);

            imgClassifier.write(imgFlipped);

            yInfo("Sending training request (for flipped image): %s",cmd.toString().c_str());
            rpcClassifier.write(cmd,reply);
            yInfo("Received reply (for flipped image): %s",reply.toString().c_str());
        }
    }

    // release resources
    mutexResources.unlock();
}


/**********************************************************/
void Manager::improve_train(const string &object, const Bottle &blobs,
                            const int i)
{
    cv::Point ref_cog=getBlobCOG(blobs,i);
    if ((ref_cog.x==RET_INVALID) || (ref_cog.y==RET_INVALID))
        return;

    double t0=Time::now();
    while (Time::now()-t0<improve_train_period)
    {
        // acquire image for training
        acquireImage();

        // grab the blobs
        Bottle blobs=getBlobs();

        // failure handling
        if (blobs.size()==0)
            continue;

        // enforce 2D consistency
        int exploredBlob=-1;
        double curMinDist=10.0;
        double curMinDist2=curMinDist*curMinDist;
        for (int i=0; i<blobs.size(); i++)
        {
            cv::Point cog=getBlobCOG(blobs,i);
            if ((cog.x==RET_INVALID) || (cog.y==RET_INVALID))
                continue;

            double dx=ref_cog.x-cog.x;
            double dy=ref_cog.y-cog.y;
            double dist2=dx*dx+dy*dy;
            if (dist2<curMinDist2)
            {
                exploredBlob=i;
                curMinDist2=dist2;
            }
        }

        // no candidate found => skip
        if (exploredBlob<0)
            continue;

        // train the classifier
        train(object,blobs,exploredBlob);

        // draw the blobs highlighting the explored one
        drawBlobs(blobs,exploredBlob);
    }
}


/**********************************************************/
Vector Manager::applyObjectPosOffsets(const string &object,
                                      const string &hand)
{
    Vector offs(3,0.0);
    if (rpcMemory.getOutputCount()>0)
    {
        mutexResourcesMemory.lock();
        map<string,int>::iterator id=memoryIds.find(object);
        map<string,int>::iterator memoryIdsEnd=memoryIds.end();
        mutexResourcesMemory.unlock(); 

        if (id!=memoryIdsEnd)
        {
            // get the relevant properties
            // [get] (("id" <num>) ("propSet" ("kinematic_offset_$hand")))
            string prop("kinematic_offset_"+hand);

            Bottle cmdMemory,replyMemory;
            cmdMemory.addVocab(Vocab::encode("get"));
            Bottle &content=cmdMemory.addList();
            Bottle &list_bid=content.addList();
            list_bid.addString("id");
            list_bid.addInt(id->second);
            Bottle &list_propSet=content.addList();
            list_propSet.addString("propSet");
            Bottle &list_items=list_propSet.addList();
            list_items.addString(prop);
            rpcMemory.write(cmdMemory,replyMemory);

            // retrieve kinematic offset
            if (replyMemory.get(0).asVocab()==Vocab::encode("ack"))
            {
                if (Bottle *propField=replyMemory.get(1).asList())
                {
                    if (propField->check(prop))
                    {
                        if (Bottle *pPos=propField->find(prop).asList())
                        {
                            if (pPos->size()>=3)
                            {
                                offs[0]=pPos->get(0).asDouble();
                                offs[1]=pPos->get(1).asDouble();
                                offs[2]=pPos->get(2).asDouble();
                            }
                        }
                    }
                }
            }
        }
    }

    return offs;
}


/**********************************************************/
void Manager::home()
{
    Vector x(3,0.0);
    x[0]=0.3; x[2]=0.6;

    Bottle b;
    b.addList().read(x);

    Property &cmd=motor.prepare();
    cmd.clear();

    cmd.put("control-frame","depth");
    cmd.put("target-type","cartesian");
    cmd.put("target-location",b.get(0));

    motor.writeStrict();

    Time::delay(4.0);
}


/**********************************************************/
void Manager::look(const string &object)
{
    Vector x;
    get3DPositionFromMemory(object,x,false);

    Bottle b;
    b.addList().read(x);

    Property &cmd=motor.prepare();
    cmd.clear();

    cmd.put("control-frame","depth");
    cmd.put("target-type","cartesian");
    cmd.put("target-location",b.get(0));

    motor.writeStrict();

    Time::delay(4.0);
}


/**********************************************************/
void Manager::look(const Bottle &blobs, const int i,
                   const Bottle &options)
{
    cv::Point cog=getBlobCOG(blobs,i);
    if ((cog.x==RET_INVALID) || (cog.y==RET_INVALID))
        return;

    Bottle b;
    Bottle &bl=b.addList();
    bl.addInt(cog.x);
    bl.addInt(cog.y);

    Property &cmd=motor.prepare();
    cmd.clear();

    cmd.put("control-frame","depth");
    cmd.put("target-type","image");
    cmd.put("image","depth_rgb");
    cmd.put("target-location",b.get(0));

    motor.writeStrict();

    Time::delay(4.0);
}


/**********************************************************/
int Manager::recognize(const string &object, Bottle &blobs,
                       Classifier **ppClassifier)
{
    map<string,Classifier*>::iterator it=db.find(object);
    if (it==db.end())
    {
        // if not, create a brand new one
        db[object]=new Classifier(object,classification_threshold);
        trackersPool[object]=Tracker(tracker_type,tracker_timeout);
        it=db.find(object);
        yInfo("created classifier for %s",object.c_str());
    }

    // acquire image for classification/training
    acquireImage();

    // grab the blobs
    blobs=getBlobs();

    // failure handling
    if (blobs.size()==0)
        return RET_INVALID;

    // get the scores from the learning machine
    Bottle scores=classify(blobs);

    // failure handling
    if (scores.size()==1)
    {
        if (scores.get(0).asString()=="failed")
        {
            speaker.speak("Ooops! Sorry, something went wrong in my brain");
            return RET_INVALID;
        }
    }

    // find the best blob
    int recogBlob=db.processScores(it->second,scores);

    // draw the blobs highlighting the recognized one (if any)
    drawBlobs(blobs,recogBlob);

    // prepare output
    if (ppClassifier!=NULL)
        *ppClassifier=it->second;

    return recogBlob;
}


/**********************************************************/
int Manager::recognize(Bottle &blobs, Bottle &scores, string &object)
{
    object=OBJECT_UNKNOWN;

    // acquire image for classification/training
    acquireImage();

    // grab the blobs
    blobs=getBlobs();

    // failure handling
    if (blobs.size()==0)
        return RET_INVALID;

    // get the scores from the learning machine
    scores=classify(blobs);

    // failure handling
    if (scores.size()==1)
    {
        if (scores.get(0).asString()=="failed")
        {
            speaker.speak("Ooops! Sorry, something went wrong in my brain");
            return RET_INVALID;
        }
    }

    // handle the human-pointed object
    if (whatGood)
    {
        int closestBlob=findClosestBlob(blobs,whatLocation);
        drawBlobs(blobs,closestBlob);
        look(blobs,closestBlob);

        ostringstream tag;
        tag<<"blob_"<<closestBlob;
        object=db.findName(scores,tag.str());
        return closestBlob;
    }
    else
    {
        speaker.speak("Ooops! Sorry, I missed where you pointed at");
        return RET_INVALID;
    }
}


/**********************************************************/
void Manager::execName(const string &object)
{
    Bottle replyHuman;
    if (!trackStopGood)
    {
        speaker.speak("Ooops! Sorry, I missed where you pointed at");
        replyHuman.addString("nack");
        rpcHuman.reply(replyHuman);
        return;
    }

    map<string,Classifier*>::iterator it=db.find(object);
    if (it==db.end())
    {
        // if not, create a brand new one
        db[object]=new Classifier(object,classification_threshold);
        trackersPool[object]=Tracker(tracker_type,tracker_timeout);
        it=db.find(object);
        yInfo("created classifier for %s",object.c_str());
    }

    // acquire image for training
    acquireImage();

    // grab the blobs
    Bottle blobs=getBlobs();

    // failure handling
    if (blobs.size()==0)
    {
        speaker.speak("Ooops! Sorry, I cannot see any object");
        replyHuman.addString("nack");
        rpcHuman.reply(replyHuman);
        return;
    }

    // run the normal procedure straightaway
    Bottle scores=classify(blobs);

    // failure handling
    if (scores.size()==1)
    {
        if (scores.get(0).asString()=="failed")
        {
            speaker.speak("Ooops! Sorry, something went wrong in my brain");
            replyHuman.addString("nack");
            rpcHuman.reply(replyHuman);
            return;
        }
    }

    db.processScores(it->second,scores);

    // find the closest blob
    int closestBlob=findClosestBlob(blobs,trackStopLocation);

    // draw the blobs highlighting the detected one (if any)
    drawBlobs(blobs,closestBlob);

    // train
    burst("start");
    train(object,blobs,closestBlob);
    improve_train(object,blobs,closestBlob);
    burst("stop");
    triggerRecogInfo(object,blobs,closestBlob,"creation");
    ostringstream reply;
    reply<<"All right! Now I know what a "<<object;
    reply<<" is";
    speaker.speak(reply.str());
    look(blobs,closestBlob);

    replyHuman.addString("ack");
    rpcHuman.reply(replyHuman);
}


/**********************************************************/
void Manager::execForget(const string &object)
{
    Bottle cmdClassifier,replyClassifier,replyHuman;

    // grab resources
    mutexResources.lock();

    // forget the whole memory
    if (object=="all")
    {
        cmdClassifier.addVocab(Vocab::encode("forget"));
        cmdClassifier.addString("all");
        yInfo("Sending clearing request: %s",cmdClassifier.toString().c_str());
        rpcClassifier.write(cmdClassifier,replyClassifier);
        yInfo("Received reply: %s",replyClassifier.toString().c_str());

        // clear the memory too
        if (rpcMemory.getOutputCount()>0)
        {
            mutexResourcesMemory.lock();
            for (map<string,int>::iterator id=memoryIds.begin(); id!=memoryIds.end(); id++)
            {
                Bottle cmdMemory,replyMemory;
                cmdMemory.addVocab(Vocab::encode("del"));
                Bottle &bid=cmdMemory.addList().addList();
                bid.addString("id");
                bid.addInt(id->second);
                rpcMemory.write(cmdMemory,replyMemory);
            }
            memoryIds.clear();
            mutexResourcesMemory.unlock();
        }

        db.clear();
        trackersPool.clear();
        speaker.speak("I have forgotten everything");
        replyHuman.addString("ack");
    }
    else    // forget specific object
    {
        ostringstream reply;
        map<string,Classifier*>::iterator it=db.find(object);
        if (it!=db.end())
        {
            cmdClassifier.addVocab(Vocab::encode("forget"));
            cmdClassifier.addString(object.c_str());
            yInfo("Sending clearing request: %s",cmdClassifier.toString().c_str());
            rpcClassifier.write(cmdClassifier,replyClassifier);
            yInfo("Received reply: %s",replyClassifier.toString().c_str());

            // remove the item from the memory too
            if (rpcMemory.getOutputCount()>0)
            {
                mutexResourcesMemory.lock();
                map<string,int>::iterator id=memoryIds.find(object);
                map<string,int>::iterator memoryIdsEnd=memoryIds.end();
                mutexResourcesMemory.unlock();

                if (id!=memoryIdsEnd)
                {
                    Bottle cmdMemory,replyMemory;
                    cmdMemory.addVocab(Vocab::encode("del"));
                    Bottle &bid=cmdMemory.addList().addList();
                    bid.addString("id");
                    bid.addInt(id->second);
                    rpcMemory.write(cmdMemory,replyMemory);

                    mutexResourcesMemory.lock();
                    memoryIds.erase(id);
                    mutexResourcesMemory.unlock();
                }
            }

            db.erase(it);
            trackersPool.erase(object);
            reply<<object<<" forgotten";
            speaker.speak(reply.str());
            replyHuman.addString("ack");
        }
        else
        {
            yInfo("%s object is unknown",object.c_str());
            reply<<"I do not know any "<<object;
            speaker.speak(reply.str());
            replyHuman.addString("nack");
        }        
    }

    rpcHuman.reply(replyHuman);

    // release resources
    mutexResources.unlock();
}


/**********************************************************/
void Manager::execWhat(const Bottle &blobs, const int pointedBlob,
                       const Bottle &scores, const string &object)
{
    Bottle cmdHuman,valHuman,replyHuman;
    Classifier *pClassifier=NULL;

    // some known object has been recognized
    if (object!=OBJECT_UNKNOWN)
    {
        ostringstream reply;
        reply<<"I think it is the "<<object;
        speaker.speak(reply.str());
        speaker.speak("Am I right?");
        yInfo("I think the blob %d is the %s",pointedBlob,object.c_str());

        // retrieve the corresponding classifier
        map<string,Classifier*>::iterator it=db.find(object);
        if (it!=db.end())
            pClassifier=it->second;

        replyHuman.addString("ack");
        replyHuman.addString(object.c_str());
    }
    // no known object has been recognized in the scene
    else
    {
        speaker.speak("I do not know this object");
        speaker.speak("What is it?");
        yInfo("No object recognized");
        replyHuman.addString("nack");
    }

    rpcHuman.reply(replyHuman);

    // enter the human interaction mode to refine the knowledge
    bool ok=false;
    while (!ok)
    {
        replyHuman.clear();
        rpcHuman.read(cmdHuman,true);

        if (isStopping())
            return;

        int type=processHumanCmd(cmdHuman,valHuman);
        // do nothing
        if (type==Vocab::encode("skip"))
        {
            speaker.speak("Skipped");
            replyHuman.addString("ack");
            ok=true;
        }
        // good job is done
        else if ((object!=OBJECT_UNKNOWN) && (type==Vocab::encode("ack")))
        {
            // reinforce if an object is available
            if (!skipLearningUponSuccess && (pointedBlob>=0) && (pClassifier!=NULL))
            {
                burst("start");
                train(object,blobs,pointedBlob);
                improve_train(object,blobs,pointedBlob);
                burst("stop");
                db.processScores(pClassifier,scores);
                pClassifier->positive();
                triggerRecogInfo(object,blobs,pointedBlob,"recognition");
                updateClassifierInMemory(pClassifier);
            }

            speaker.speak("Cool!");
            replyHuman.addString("ack");
            ok=true;
        }
        // misrecognition
        else if (type==Vocab::encode("nack"))
        {
            // update the threshold
            if ((pointedBlob>=0) && (pClassifier!=NULL))
            {
                db.processScores(pClassifier,scores);
                pClassifier->negative();
                updateClassifierInMemory(pClassifier);
            }

            speaker.speak("Sorry");
            replyHuman.addString("ack");
            ok=true;
        }
        // handle new/unrecognized/misrecognized object
        else if ((type==Vocab::encode("name")) && (valHuman.size()>0))
        {
            string objectName=valHuman.get(0).asString().c_str();

            // check whether the object is already known
            // and, if not, allocate space for it
            map<string,Classifier*>::iterator it=db.find(objectName);
            if (it==db.end())
            {
                db[objectName]=new Classifier(objectName,classification_threshold);
                trackersPool[objectName]=Tracker(tracker_type,tracker_timeout);
                it=db.find(objectName);
                speaker.speak("Oooh, I see");
                yInfo("created classifier for %s",objectName.c_str());
            }
            else
            {
                // update the threshold for the case of misrecognition
                if ((pClassifier!=NULL) && (object!=objectName) && (object!=OBJECT_UNKNOWN))
                {
                    db.processScores(pClassifier,scores);
                    pClassifier->negative();
                    updateClassifierInMemory(pClassifier);
                }

                ostringstream reply;
                reply<<"Sorry, I should have recognized the "<<objectName;
                speaker.speak(reply.str());
            }

            // trigger the classifier
            if (pointedBlob>=0)
            {
                burst("start");
                train(objectName,blobs,pointedBlob);
                improve_train(objectName,blobs,pointedBlob);
                burst("stop");
                triggerRecogInfo(objectName,blobs,pointedBlob,
                                 (object==OBJECT_UNKNOWN)?"creation":"recognition");
            }

            db.processScores(it->second,scores);

            replyHuman.addString("ack");
            ok=true;
        }
        else
        {
            speaker.speak("Hmmm hmmm hmmm! Try again");
            replyHuman.addString("nack");
        }

        rpcHuman.reply(replyHuman);
    }
}


/**********************************************************/
void Manager::execThis(const string &object, const string &detectedObject,
                       const Bottle &blobs, const int &pointedBlob)
{
    if (pointedBlob>=0)
    {
        Bottle replyHuman;

        string recogType="recognition";
        map<string,Classifier*>::iterator it=db.find(object);
        if (it==db.end())
        {
            // if not, create a brand new one
            db[object]=new Classifier(object,classification_threshold);
            trackersPool[object]=Tracker(tracker_type,tracker_timeout);
            it=db.find(object);
            yInfo("created classifier for %s",object.c_str());
            recogType="creation";
        }

        ostringstream reply;

        //if the classifier recognized the object
        if (object.compare(detectedObject)==0)
            reply<<"Yes, I know that is a "<<object<<"!";
        else if(detectedObject.compare(OBJECT_UNKNOWN)==0)
            reply<<"All right! Now I know what a "<<object<<" is!";
        else
        {
            reply<<"Oh dear, I thought that was a "<<detectedObject<<"?";
       
            map<string,Classifier*>::iterator it_detected=db.find(detectedObject);
            if (it_detected==db.end())
            {
                it_detected->second->negative();
                updateClassifierInMemory(it_detected->second);
            }
        }

        burst("start");
        train(object,blobs,pointedBlob);
        improve_train(object,blobs,pointedBlob);
        burst("stop");
 
        speaker.speak(reply.str());
        look(blobs,pointedBlob);    

        replyHuman.addString("ack");
        rpcHuman.reply(replyHuman);
    }
}


/**********************************************************/
void Manager::switchAttention()
{
    // skip if connection with motor interface is not in place
    if (motor.getOutputCount()>0)
    {
        LockGuard lg(mutexAttention);

        // grab the blobs
        Bottle blobs=getBlobs();
        for (int i=0; i<blobs.size(); i++)
        {
            // make a guess
            int guess=(int)Rand::scalar(0.0,blobs.size());
            if (guess>=blobs.size())
                guess=blobs.size()-1;

            cv::Point cog=getBlobCOG(blobs,guess);
            if ((cog.x==RET_INVALID) || (cog.y==RET_INVALID))
                continue;

            look(blobs,guess);
            return;
        }

        // if no good blob found go home
        home();
    }
}


/**********************************************************/
void Manager::doLocalization()
{
    // acquire image for classification/training
    acquireImage(true);
    // grab the blobs
    Bottle blobs=getBlobs();
    // get the scores from the learning machine
    Bottle scores=classify(blobs,true);
    // update location of histogram display
    if (Bottle *loc=histObjLocPort.read(false))
    {        
        if (loc->size()>=2)
        {
            Vector x;
            clickLocation=cv::Point(loc->get(0).asInt(),loc->get(1).asInt());
            if (get3DPosition(clickLocation,x))
                histObjLocation=x;
        }
    }
    // find the closest blob to the location of histogram display
    int closestBlob=findClosestBlob(blobs,histObjLocation);
    // draw the blobs
    drawBlobs(blobs,closestBlob,&scores);
    // draw scores histogram
    drawScoresHistogram(blobs,scores,closestBlob);

    // data for memory update
    mutexResourcesMemory.lock();
    memoryBlobs=blobs;
    memoryScores=scores;
    mutexResourcesMemory.unlock();
}


/**********************************************************/
bool Manager::get3DPositionFromMemory(const string &object,
                                      Vector &position,
                                      const bool lockMemory)
{
    bool ret=false;
    if (rpcMemory.getOutputCount()>0)
    {
        // grab resources
        if (lockMemory)
            LockGuard lg(mutexMemoryUpdate); 

        mutexResourcesMemory.lock();
        map<string,int>::iterator id=memoryIds.find(object);
        map<string,int>::iterator memoryIdsEnd=memoryIds.end();
        mutexResourcesMemory.unlock(); 

        if (id!=memoryIdsEnd)
        {
            // get the relevant properties
            // [get] (("id" <num>) ("propSet" ("position_3d")))
            Bottle cmdMemory,replyMemory;
            cmdMemory.addVocab(Vocab::encode("get"));
            Bottle &content=cmdMemory.addList();
            Bottle &list_bid=content.addList();
            list_bid.addString("id");
            list_bid.addInt(id->second);
            Bottle &list_propSet=content.addList();
            list_propSet.addString("propSet");
            Bottle &list_items=list_propSet.addList();
            list_items.addString("position_3d");
            rpcMemory.write(cmdMemory,replyMemory);

            // retrieve 3D position
            if (replyMemory.get(0).asVocab()==Vocab::encode("ack"))
            {
                if (Bottle *propField=replyMemory.get(1).asList())
                {
                    if (propField->check("position_3d"))
                    {
                        if (Bottle *pPos=propField->find("position_3d").asList())
                        {
                            if (pPos->size()>=3)
                            {
                                position.resize(3);
                                position[0]=pPos->get(0).asDouble();
                                position[1]=pPos->get(1).asDouble();
                                position[2]=pPos->get(2).asDouble();
                                ret=true;
                            }
                        }
                    }
                }
            }
        }
    }

    return ret;
}


/**********************************************************/
void Manager::updateMemory()
{
    if (rpcMemory.getOutputCount()>0)
    {
        // grab resources
        mutexMemoryUpdate.lock();

        // load memory on connection event
        if (scheduleLoadMemory)
        {
            loadMemory();
            scheduleLoadMemory=false;
        }

        // latch image
        ImageOf<PixelBgr> &imgLatch=imgTrackOut.prepare();
        cv::Mat imgLatchMat=cv::cvarrToMat(imgLatch.getIplImage());

        mutexResourcesMemory.lock();
        Bottle blobs=memoryBlobs;
        Bottle scores=memoryScores;
        imgLatch=imgRtLoc;
        mutexResourcesMemory.unlock();

        // reset internal tracking state
        for (map<string,Tracker>::iterator it=trackersPool.begin(); it!=trackersPool.end(); it++)
            it->second.prepare();
        
        for (int j=0; j<blobs.size(); j++)
        {
            Bottle *item=blobs.get(j).asList();
            if (item==NULL)
                continue;

            ostringstream tag;
            tag<<"blob_"<<j;

            // find the blob name (or unknown)
            mutexResources.lock();
            string object=db.findName(scores,tag.str());
            mutexResources.unlock();

            if (object!=OBJECT_UNKNOWN)
            {
                // compute the bounding box
                cv::Point tl,br;
                tl.x=(int)item->get(0).asDouble();
                tl.y=(int)item->get(1).asDouble();
                br.x=(int)item->get(2).asDouble();
                br.y=(int)item->get(3).asDouble();

                cv::Rect bbox(tl.x,tl.y,br.x-tl.x,br.y-tl.y);
                if (thresBBox(bbox,imgLatch))
                {
                    map<string,Tracker>::iterator tracker=trackersPool.find(object);
                    if (tracker!=trackersPool.end())
                        tracker->second.latchBBox(bbox);
                }
            }
        }

        // cycle over objects to handle tracking
        set<int> avalObjIds;
        for (map<string,Tracker>::iterator it=trackersPool.begin(); it!=trackersPool.end(); it++)
        {
            string object=it->first;
            it->second.track(imgLatch);

            cv::Rect bbox;
            if (it->second.is_tracking(bbox))
            {
                // threshold bbox
                if (!thresBBox(bbox,imgLatch))
                    continue;

                cv::Point tl(bbox.x,bbox.y);
                cv::Point br(bbox.x+bbox.width,bbox.y+bbox.height);
                cv::Point cog((tl.x+br.x)>>1,(tl.y+br.y)>>1);
                Vector x;

                // find 3d position
                if (get3DPosition(cog,x))
                {
                    // prepare position_2d property
                    Bottle position_2d;
                    Bottle &list_2d=position_2d.addList();
                    list_2d.addString("position_2d");
                    Bottle &list_2d_c=list_2d.addList();
                    list_2d_c.addDouble(tl.x);
                    list_2d_c.addDouble(tl.y);
                    list_2d_c.addDouble(br.x);
                    list_2d_c.addDouble(br.y);

                    // prepare position_3d property
                    Bottle position_3d;
                    Bottle &list_3d=position_3d.addList();
                    list_3d.addString("position_3d");
                    list_3d.addList().read(x);

                    mutexResourcesMemory.lock();
                    map<string,int>::iterator id=memoryIds.find(object);
                    map<string,int>::iterator memoryIdsEnd=memoryIds.end();
                    mutexResourcesMemory.unlock();

                    Bottle cmdMemory,replyMemory;
                    if (id==memoryIdsEnd)      // the object is not available => [add]
                    {
                        cmdMemory.addVocab(Vocab::encode("add"));
                        Bottle &content=cmdMemory.addList();
                        Bottle &list_entity=content.addList();
                        list_entity.addString("entity");
                        list_entity.addString("object");
                        Bottle &list_name=content.addList();
                        list_name.addString("name");
                        list_name.addString(object.c_str());
                        content.append(position_2d);
                        content.append(position_3d);
                        rpcMemory.write(cmdMemory,replyMemory);

                        if (replyMemory.size()>1)
                        {
                            // store the id for later usage
                            if (replyMemory.get(0).asVocab()==Vocab::encode("ack"))
                            {
                                if (Bottle *idField=replyMemory.get(1).asList())
                                {
                                    int id=idField->get(1).asInt();
                                    mutexResourcesMemory.lock();
                                    memoryIds[object]=id;
                                    mutexResourcesMemory.unlock();

                                    avalObjIds.insert(id);
                                }
                                else
                                    continue;
                            }
                        }
                    }
                    else    // the object is already available => [set]
                    {
                        // prepare id property
                        Bottle bid;
                        Bottle &list_bid=bid.addList();
                        list_bid.addString("id");
                        list_bid.addInt(id->second);

                        cmdMemory.addVocab(Vocab::encode("set"));
                        Bottle &content=cmdMemory.addList();
                        content.append(bid);
                        content.append(position_2d);
                        content.append(position_3d);
                        rpcMemory.write(cmdMemory,replyMemory);

                        avalObjIds.insert(id->second);
                    }

                    // highlight location of tracked blobs within images
                    cv::rectangle(imgLatchMat,tl,br,cv::Scalar(255,0,0),2);
                    cv::putText(imgLatchMat,object.c_str(),cv::Point(tl.x,tl.y-5),
                                cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,0,0),2);
                }
            }
        }

        if (imgTrackOut.getOutputCount()>0)
            imgTrackOut.write();
        else
            imgTrackOut.unprepare();

        // remove position properties of objects not in scene
        mutexResourcesMemory.lock();
        for (map<string,int>::iterator it=memoryIds.begin(); it!=memoryIds.end(); it++)
        {
            int id=it->second;
            if (avalObjIds.find(id)==avalObjIds.end())
            {
                Bottle cmdMemory,replyMemory;
                cmdMemory.addVocab(Vocab::encode("del"));
                Bottle &content=cmdMemory.addList();
                Bottle &list_bid=content.addList();
                list_bid.addString("id");
                list_bid.addInt(id);
                Bottle &list_propSet=content.addList();
                list_propSet.addString("propSet");
                Bottle &list_items=list_propSet.addList();
                list_items.addString("position_2d");
                list_items.addString("position_3d");
                rpcMemory.write(cmdMemory,replyMemory);
            }
        }
        mutexResourcesMemory.unlock();

        // release resources
        mutexMemoryUpdate.unlock();
    }
}


/**********************************************************/
void Manager::updateClassifierInMemory(Classifier *pClassifier)
{
    if ((rpcMemory.getOutputCount()>0) && (pClassifier!=NULL))
    {
        string objectName=pClassifier->getName();

        // prepare classifier_thresholds property
        Bottle classifier_property;
        Bottle &list_classifier=classifier_property.addList();
        list_classifier.addString("classifier_thresholds");
        list_classifier.addList().append(pClassifier->toBottle());

        mutexResourcesMemory.lock();
        map<string,int>::iterator id=memoryIds.find(objectName);
        map<string,int>::iterator memoryIdsEnd=memoryIds.end();
        mutexResourcesMemory.unlock();

        Bottle cmdMemory,replyMemory;
        if (id==memoryIdsEnd)      // the object is not available => [add]
        {
            cmdMemory.addVocab(Vocab::encode("add"));
            Bottle &content=cmdMemory.addList();
            Bottle &list_entity=content.addList();
            list_entity.addString("entity");
            list_entity.addString("object");
            Bottle &list_name=content.addList();
            list_name.addString("name");
            list_name.addString(objectName.c_str());
            content.append(classifier_property);
            rpcMemory.write(cmdMemory,replyMemory);

            if (replyMemory.size()>1)
            {
                // store the id for later usage
                if (replyMemory.get(0).asVocab()==Vocab::encode("ack"))
                {
                    if (Bottle *idField=replyMemory.get(1).asList())
                    {
                        mutexResourcesMemory.lock();
                        memoryIds[objectName]=idField->get(1).asInt();
                        mutexResourcesMemory.unlock();
                    }
                }
            }
        }
        else    // the object is already available => [set]
        {
            // prepare id property
            Bottle bid;
            Bottle &list_bid=bid.addList();
            list_bid.addString("id");
            list_bid.addInt(id->second);

            cmdMemory.addVocab(Vocab::encode("set"));
            Bottle &content=cmdMemory.addList();
            content.append(bid);
            content.append(classifier_property);
            rpcMemory.write(cmdMemory,replyMemory);
        }
    }
}


/**********************************************************/
Vector Manager::updateObjCartPosInMemory(const string &object, 
                                         const Bottle &blobs,
                                         const int i)
{
    Vector x(3,0.0);
    if ((rpcMemory.getOutputCount()>0) && (i!=RET_INVALID) && (i<blobs.size()))
    {
        mutexResourcesMemory.lock();
        map<string,int>::iterator id=memoryIds.find(object);
        map<string,int>::iterator memoryIdsEnd=memoryIds.end();
        mutexResourcesMemory.unlock();

        Bottle *item=blobs.get(i).asList();
        if ((id!=memoryIdsEnd) && (item!=NULL))
        {
            cv::Point cog=getBlobCOG(blobs,i);
            if ((cog.x==RET_INVALID) || (cog.y==RET_INVALID))
                return x;

            if (get3DPosition(cog,x))
            {
                Bottle cmdMemory,replyMemory;

                // prepare id property
                Bottle bid;
                Bottle &list_bid=bid.addList();
                list_bid.addString("id");
                list_bid.addInt(id->second);

                // prepare position_2d property
                Bottle position_2d;
                Bottle &list_2d=position_2d.addList();
                list_2d.addString("position_2d");
                Bottle &list_2d_c=list_2d.addList();
                list_2d_c.addDouble(item->get(0).asDouble());
                list_2d_c.addDouble(item->get(1).asDouble());
                list_2d_c.addDouble(item->get(2).asDouble());
                list_2d_c.addDouble(item->get(3).asDouble());

                // prepare position_3d property
                Bottle position_3d;
                Bottle &list_3d=position_3d.addList();
                list_3d.addString("position_3d");
                list_3d.addList().read(x);

                cmdMemory.addVocab(Vocab::encode("set"));
                Bottle &content=cmdMemory.addList();
                content.append(bid);
                content.append(position_2d);
                content.append(position_3d);
                rpcMemory.write(cmdMemory,replyMemory);
            }
        }
    }

    return x;
}


/**********************************************************/
void Manager::triggerRecogInfo(const string &object, const Bottle &blobs,
                               const int i, const string &recogType)
{
    if ((recogTriggerPort.getOutputCount()>0) && (i!=RET_INVALID) && (i<blobs.size()))
    {
        cv::Point cog=getBlobCOG(blobs,i);
        if ((cog.x==RET_INVALID) || (cog.y==RET_INVALID))
            return;

        Vector x;
        if (get3DPosition(cog,x))
        {
            Property &msg=recogTriggerPort.prepare();
            msg.clear();

            Bottle pos; pos.addList().read(x);
            msg.put("label",object.c_str());
            msg.put("position_3d",pos.get(0));
            msg.put("type",recogType.c_str());

            recogTriggerPort.write();
        }
    }
}


/**********************************************************/
void Manager::loadMemory()
{
    yInfo("Loading memory ...");
    // grab resources
    mutexResourcesMemory.lock();

    // purge internal databases
    memoryIds.clear();
    db.clear();
    trackersPool.clear();

    // ask for all the items stored in memory
    Bottle cmdMemory,replyMemory,replyMemoryProp;
    cmdMemory.addVocab(Vocab::encode("ask"));
    Bottle &content=cmdMemory.addList().addList();
    content.addString("entity");
    content.addString("==");
    content.addString("object");
    rpcMemory.write(cmdMemory,replyMemory);
    
    if (replyMemory.size()>1)
    {
        if (replyMemory.get(0).asVocab()==Vocab::encode("ack"))
        {
            if (Bottle *idField=replyMemory.get(1).asList())
            {
                if (Bottle *idValues=idField->get(1).asList())
                {
                    // cycle over items
                    for (int i=0; i<idValues->size(); i++)
                    {
                        int id=idValues->get(i).asInt();

                        // get the relevant properties
                        // [get] (("id" <num>) ("propSet" ("name" "classifier_thresholds")))
                        cmdMemory.clear();
                        cmdMemory.addVocab(Vocab::encode("get"));
                        Bottle &content=cmdMemory.addList();
                        Bottle &list_bid=content.addList();
                        list_bid.addString("id");
                        list_bid.addInt(id);
                        Bottle &list_propSet=content.addList();
                        list_propSet.addString("propSet");
                        Bottle &list_items=list_propSet.addList();
                        list_items.addString("name");
                        list_items.addString("classifier_thresholds");
                        rpcMemory.write(cmdMemory,replyMemoryProp);

                        // update internal databases
                        if (replyMemoryProp.get(0).asVocab()==Vocab::encode("ack"))
                        {
                            if (Bottle *propField=replyMemoryProp.get(1).asList())
                            {
                                if (propField->check("name"))
                                {
                                    string object=propField->find("name").asString().c_str();
                                    memoryIds[object]=id;

                                    if (propField->check("classifier_thresholds"))
                                        db[object]=new Classifier(*propField->find("classifier_thresholds").asList());
                                    else
                                        db[object]=new Classifier(object,classification_threshold);
                                    trackersPool[object]=Tracker(tracker_type,tracker_timeout);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    yInfo("Objects in memory: %d",(int)db.size());
    for (map<string,Classifier*>::iterator it=db.begin(); it!=db.end(); it++)
    {
        string object=it->first;
        string properties=it->second->toBottle().toString().c_str();
        yInfo("classifier for %s: memory_id=%d; properties=%s",
              object.c_str(),memoryIds[object],properties.c_str());
    }

    // release resources
    mutexResourcesMemory.unlock();
    yInfo("Memory loaded");
}


/**********************************************************/
bool Manager::configure(ResourceFinder &rf)
{
    name=rf.check("name",Value("iolStateMachineHandler")).asString().c_str();

    imgIn.open(("/"+name+"/img:i").c_str());
    blobExtractor.open(("/"+name+"/blobs:i").c_str());
    imgOut.open(("/"+name+"/img:o").c_str());
    imgRtLocOut.open(("/"+name+"/imgLoc:o").c_str());
    imgTrackOut.open(("/"+name+"/imgTrack:o").c_str());
    imgClassifier.open(("/"+name+"/imgClassifier:o").c_str());
    imgHistogram.open(("/"+name+"/imgHistogram:o").c_str());
    histObjLocPort.open(("/"+name+"/histObjLocation:i").c_str());
    recogTriggerPort.open(("/"+name+"/recog:o").c_str());

    cerGazePort.open(("/"+name+"/gaze:i").c_str());

    rpcPort.open(("/"+name+"/rpc").c_str());
    rpcHuman.open(("/"+name+"/human:rpc").c_str());
    rpcClassifier.open(("/"+name+"/classify:rpc").c_str());
    rpcMotorGrasp.open(("/"+name+"/motor_grasp:rpc").c_str());
    rpcGet3D.open(("/"+name+"/get3d:rpc").c_str());
    motor.open(("/"+name+"/motor:o").c_str());

    pointedLoc.open(("/"+name+"/point:i").c_str());
    speaker.open(("/"+name+"/speak:o").c_str());

    memoryReporter.setManager(this);
    rpcMemory.setReporter(memoryReporter);
    rpcMemory.open(("/"+name+"/memory:rpc").c_str());

    skim_blobs_x_bounds.resize(2);
    skim_blobs_x_bounds[0]=-0.50;
    skim_blobs_x_bounds[1]=-0.10;
    if (rf.check("skim_blobs_x_bounds"))
    {
        if (Bottle *bounds=rf.find("skim_blobs_x_bounds").asList())
        {
            if (bounds->size()>=2)
            {
                skim_blobs_x_bounds[0]=bounds->get(0).asDouble();
                skim_blobs_x_bounds[1]=bounds->get(1).asDouble();
            }
        }
    }

    skim_blobs_y_bounds.resize(2);
    skim_blobs_y_bounds[0]=-0.30;
    skim_blobs_y_bounds[1]=+0.30;
    if (rf.check("skim_blobs_y_bounds"))
    {
        if (Bottle *bounds=rf.find("skim_blobs_y_bounds").asList())
        {
            if (bounds->size()>=2)
            {
                skim_blobs_y_bounds[0]=bounds->get(0).asDouble();
                skim_blobs_y_bounds[1]=bounds->get(1).asDouble();
            }
        }
    }

    // location used to display the
    // histograms upon the closest blob
    clickLocation=cv::Point(0,0);
    histObjLocation.resize(3);
    histObjLocation[0]=-0.3;
    histObjLocation[1]=0.0;
    histObjLocation[2]=-0.1;
    
    attention.setManager(this);
    attention.start();

    doAttention=rf.check("attention",Value("on")).asString()=="on";
    if (!doAttention)
        attention.suspend();

    lastBlobsArrivalTime=0.0;
    rtLocalization.setManager(this);
    rtLocalization.setRate(rf.check("rt_localization_period",Value(30)).asInt());
    rtLocalization.start();

    memoryUpdater.setManager(this);
    memoryUpdater.setRate(rf.check("memory_update_period",Value(60)).asInt());
    memoryUpdater.start();

    blobs_detection_timeout=rf.check("blobs_detection_timeout",Value(0.2)).asDouble();
    improve_train_period=rf.check("improve_train_period",Value(0.0)).asDouble();
    trainOnFlipped=rf.check("train_flipped_images",Value("off")).asString()=="on";
    trainBurst=rf.check("train_burst_images",Value("off")).asString()=="on";
    skipLearningUponSuccess=rf.check("skip_learning_upon_success",Value("off")).asString()=="on";
    classification_threshold=rf.check("classification_threshold",Value(0.5)).asDouble();
    tracker_type=rf.check("tracker_type",Value("BOOSTING")).asString().c_str();
    tracker_timeout=std::max(0.0,rf.check("tracker_timeout",Value(5.0)).asDouble());

    tracker_min_blob_size.resize(2,0);
    if (rf.check("tracker_min_blob_size"))
    {
        if (Bottle *size=rf.find("tracker_min_blob_size").asList())
        {
            if (size->size()>=2)
            {
                tracker_min_blob_size[0]=size->get(0).asInt();
                tracker_min_blob_size[1]=size->get(1).asInt();
            }
        }
    }

    histFilterLength=std::max(1,rf.check("hist_filter_length",Value(10)).asInt());
    blockEyes=rf.check("block_eyes",Value(-1.0)).asDouble();    

    img.resize(320,240);
    imgRtLoc.resize(320,240);
    img.zero();
    imgRtLoc.zero();

    attach(rpcPort);
    Rand::init();

    busy=false;
    scheduleLoadMemory=false;
    enableInterrupt=false;
    trackStopGood=false;
    whatGood=false;
    skipGazeHoming=false;    

    objectToBeKinCalibrated="";

    histColorsCode.push_back(cv::Scalar( 65, 47,213));
    histColorsCode.push_back(cv::Scalar(122, 79, 58));
    histColorsCode.push_back(cv::Scalar(154,208, 72));
    histColorsCode.push_back(cv::Scalar( 71,196,249));
    histColorsCode.push_back(cv::Scalar(224,176, 96));
    histColorsCode.push_back(cv::Scalar( 22,118,238));

    return true;
}


/**********************************************************/
bool Manager::interruptModule()
{
    imgIn.interrupt();
    imgOut.interrupt();
    imgRtLocOut.interrupt();
    imgTrackOut.interrupt();
    imgClassifier.interrupt();
    imgHistogram.interrupt();
    histObjLocPort.interrupt();
    recogTriggerPort.interrupt();
    rpcPort.interrupt();
    rpcHuman.interrupt();
    blobExtractor.interrupt();
    rpcClassifier.interrupt();
    rpcMotorGrasp.interrupt();
    rpcGet3D.interrupt();
    pointedLoc.interrupt();
    speaker.interrupt();
    rpcMemory.interrupt();
    motor.interrupt();

    rtLocalization.stop();
    memoryUpdater.stop();
    attention.stop();

    return true;
}


/**********************************************************/
bool Manager::close()
{
    imgIn.close();
    imgOut.close();
    imgRtLocOut.close();
    imgTrackOut.close();
    imgClassifier.close();
    imgHistogram.close();
    histObjLocPort.close();
    recogTriggerPort.close();
    rpcPort.close();
    rpcHuman.close();
    blobExtractor.close();
    rpcClassifier.close();
    rpcMotorGrasp.close();
    rpcGet3D.close();
    pointedLoc.close();
    speaker.close();
    rpcMemory.close();
    motor.close();
    cerGazePort.close();

    // dispose filters used for scores histogram
    for (map<string,Filter*>::iterator it=histFiltersPool.begin();
         it!=histFiltersPool.end(); it++)
        delete it->second;

    return true;
}


/**********************************************************/
bool Manager::updateModule()
{
    Bottle cmdHuman,valHuman,replyHuman;
    rpcHuman.read(cmdHuman,true);

    BusyGate busyGate(busy);

    if (isStopping())
        return false;

    attention.suspend();

    int rxCmd=processHumanCmd(cmdHuman,valHuman);
    if ((rxCmd==Vocab::encode("attention")) && (valHuman.size()>0))
        if (valHuman.get(0).asString()=="stop")
            skipGazeHoming=true;

    if (!skipGazeHoming)
    {
        home();

        // this wait-state gives the memory
        // time to be updated with the 3D
        // location of the objects
        Time::delay(0.1);
    }

    skipGazeHoming=false;

    if (rxCmd==Vocab::encode("home"))
    {
        home();
        replyHuman.addString("ack");
        rpcHuman.reply(replyHuman);
    }
    else if ((rxCmd==Vocab::encode("name")) && (valHuman.size()>0))
    {        
        string activeObject=valHuman.get(0).asString().c_str();
        execName(activeObject);
    }
    else if ((rxCmd==Vocab::encode("forget")) && (valHuman.size()>0))
    {        
        string activeObject=valHuman.get(0).asString().c_str();

        mutexMemoryUpdate.lock();
        execForget(activeObject);
        mutexMemoryUpdate.unlock();
    }
    else if (rxCmd==Vocab::encode("what"))
    {
        // avoid being distracted by the human hand
        // while it is being removed: save the last
        // pointed object
        whatGood=pointedLoc.getLoc(whatLocation);
        Time::delay(1.0);

        Bottle blobs,scores;
        string activeObject;
        int pointedBlob=recognize(blobs,scores,activeObject);
        execWhat(blobs,pointedBlob,scores,activeObject);
    }
    else if ((rxCmd==Vocab::encode("this")) && (valHuman.size()>0))
    {
        // name of the object to be learned
        string activeObject=valHuman.get(0).asString().c_str();

        // get location from a click on the viewer
        if (valHuman.size()>=2)
        {
            if (valHuman.get(1).asString()=="click")
            {
                whatLocation=clickLocation;
                whatGood=true;
            }
            else
                whatGood=false;
        }
        // get location via pointing action
        else
        {
            whatGood=pointedLoc.getLoc(whatLocation); 
            Time::delay(1.0);
        }

        mutexMemoryUpdate.lock();
        Bottle blobs,scores;
        string detectedObject;
        int pointedBlob=recognize(blobs,scores,detectedObject);

        execThis(activeObject,detectedObject,blobs,pointedBlob);
        updateObjCartPosInMemory(activeObject,blobs,pointedBlob);
        mutexMemoryUpdate.unlock();
    }
    else if ((rxCmd==Vocab::encode("attention")) && (valHuman.size()>0))
    {
        string type=valHuman.get(0).asString().c_str();
        if (type=="stop")
        {
            doAttention=false;
            replyHuman.addString("ack");
        }
        else if (type=="start")
        {
            doAttention=true;
            replyHuman.addString("ack");
        }
        else
            replyHuman.addString("nack");

        rpcHuman.reply(replyHuman);
    }
    else if ((rxCmd==Vocab::encode("say")) && (valHuman.size()>0))
    {
        string speech=valHuman.get(0).asString().c_str();
        speaker.speak(speech);
        replyHuman.addString("ack");
        rpcHuman.reply(replyHuman);
        skipGazeHoming=true;
        return true; // avoid resuming the attention
    }
    else    // manage an unknown request
    {
        speaker.speak("I don't understand what you want me to do");
        replyHuman.addString("nack");
        rpcHuman.reply(replyHuman);
    }

    if (doAttention)
        attention.resume();

    return true;
}


/**********************************************************/
bool Manager::respond(const Bottle &command, Bottle &reply)
{
    string ack="ack";
    string nack="nack";
    Value cmd=command.get(0);

    string ans=nack; string pl;
    if (cmd.isVocab())
    {
        if (cmd.asVocab()==Vocab::encode("status"))
        {
            ans=ack;
            pl=busy?"busy":"idle";
        }
    }

    Bottle rep;
    if (ans==ack)
    {
        reply.addString(ack.c_str());
        reply.addString(pl.c_str());
    }
    else if (RFModule::respond(command,rep))
        reply=rep;
    else
        reply.addString(nack.c_str());

    return true;
}


/**********************************************************/
double Manager::getPeriod()
{
    // the updateModule goes through a
    // blocking read => no need for periodicity
    return 0.0;
}



