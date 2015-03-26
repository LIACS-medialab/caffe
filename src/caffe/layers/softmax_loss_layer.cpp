/* 
// Yu: This layer is to compute the training loss cost with softmax algorithm.
*/
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
/*   
// This function is to configure this layer with bottom and top blobs.
*/
template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  /* First, call the loss layer and configure it. */
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  /* read the layer parameters in the train.prototxt, and set them to softmax_param. */
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  /* create a new softmax layer */
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  /* bottom[0] is the prob values. These codes are to compute softmax for prob,
  // for example the exp process. The pro_ is the result.
  */
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]); //
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  /* If there is ignored label, then doesnot compute its loss. */
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  
  /* normlize the loss cost among batches, spatial locations, 
  // see details in caffe.proto loss_param definition.
  */
  normalize_ = this->layer_param_.loss_param().normalize();
}

/*   
// TBD.
*/
template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

/*   
// Forward process for softmax loss computation.
*/
template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  /* prob_ is the result after softmax computation. */
  const Dtype* prob_data = prob_.cpu_data();
  /* bottom[1] consists of ground truth labels, bottom[0] includes the prob values. */
  const Dtype* label = bottom[1]->cpu_data();
  /* how many batches ?! */
  int num = prob_.num();
  /* prob_.count() returns the all values in all batches. dim is the numbers of all values in one batch. */
  /* spatial_dim is the spatial size of one map. */
  int dim = prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();
  int count = 0;
  Dtype loss = 0;
  /* For each batch */ 
  for (int i = 0; i < num; ++i) {
    /* For each spatial location, compare their values in all maps. */
    for (int j = 0; j < spatial_dim; j++) {
      /* label blob has one map per batch. So understand the label index! */
      const int label_value = static_cast<int>(label[i * spatial_dim + j]);
      /* donot compute loss for ignored label. */
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      /* If label value > 0. */
      DCHECK_GE(label_value, 0);
      /* If label value < channels, such as 1000. */
      DCHECK_LT(label_value, prob_.channels());
      
      /* compute loss cost. What is FLT_MIN ? */
      loss -= log(std::max(prob_data[i * dim + label_value * spatial_dim + j],
                           Dtype(FLT_MIN)));
      
      /* count every computation. */
      ++count;
    }
  }
  /* If true, normalize the final loss among all data. 
  // else, just normalize the loss among batches.
  // What does this flag matter ? 
  */
  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = loss / count;
  } else {
    top[0]->mutable_cpu_data()[0] = loss / num;
  }
  /* all top blobs are the same. */
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

/*   
// Backward process for softmax loss computation.
*/
template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  /* TBD ?  what is propagate_down ? */
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    /* diff presents the gradients. */
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    /* copy the prob_data to bottom_diff. */
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int num = prob_.num(); // batch number
    int dim = prob_.count() / num; // all values in one batch
    int spatial_dim = prob_.height() * prob_.width();
    int count = 0;
    /* For each batch. */
    for (int i = 0; i < num; ++i) {
      /* For each spatial location. */
      for (int j = 0; j < spatial_dim; ++j) {
        const int label_value = static_cast<int>(label[i * spatial_dim + j]);
        /* For ignore labels, set their gradients with 0. 
        // Else, for normal labels, minus their gradients with 1.
        */
        if (has_ignore_label_ && label_value == ignore_label_) { 
          for (int c = 0; c < bottom[0]->channels(); ++c) {
            bottom_diff[i * dim + c * spatial_dim + j] = 0;
          }
        } else {
          /* gradient = 1- p ! */
          bottom_diff[i * dim + label_value * spatial_dim + j] -= 1; 
          ++count;
        }
      }
    }
    /* What is loss_weight ?  is the sum loss ? */
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_scal(prob_.count(), loss_weight / num, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLoss);

}  // namespace caffe
