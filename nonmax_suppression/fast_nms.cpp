/**
 * Fast non-maximum suppression in C, port from  
 * http://quantombone.blogspot.com/2011/08/blazing-fast-nmsm-from-exemplar-svm.html
 *
 * @blackball (bugway@gmail.com)
 */

#include <opencv2/core/core.hpp>
#include <vector>
#include <algorithm>
#include <iostream>

// Detected rectangle. Tracks a rectangle, its relative score and whether 
// or not it has been filtered out as a non-maximum version of a better DRect
// close by.
class DRect {
   public :
      // Constructor given an upper left and lower right coordinate pair
      DRect(int x0, int y0, int x1, int y1, float _score) :
         rect(cv::Point(x0, y0), cv::Point(x1+1, y1+1)),
         score(_score),
         keep(true)
      {
      }
      DRect(const cv::Rect &_rect, float _score) :
         rect(_rect),
         score(_score),
         keep(true)
      {
      }

      // Used for testing overlap between near-by rectangles
      float invArea(void) const
      {
         return 1.0f / rect.area();
      }

      // Getters for rectangle coords
      int x0(void) const
      {
         return rect.x;
      }
      int y0(void) const
      {
         return rect.y;
      }
      int x1(void) const
      {
         return rect.x + rect.width - 1;
      }
      int y1(void) const
      {
         return rect.y + rect.height - 1;
      }

      // Mark as invalid - this is a non-maximum version
      // of another, better DRect
      void invalidate(void)
      {
         keep = false;
      }
      bool valid(void)
      {
         return keep;
      }

      // Debugging stuff
      void print(void) const
      {
         std::cout << "x0 = " << this->x0();
         std::cout << " y0 = " << this->y0();
         std::cout << " x1 = " << this->x1();
         std::cout << " y1 = " << this->y1();
         std::cout << " score = " << score;
         std::cout << " valid = " << keep;
         std::cout << std::endl;
      }

      // Helper functions for std::sort
      bool operator< (const DRect &rhs) const
      {
         return score < rhs.score;
      }
      bool operator> (const DRect &rhs) const
      {
         return score > rhs.score;
      }
   private:
      cv::Rect rect;  // detected rectangle coordinates
      float    score; // detection score
      bool     keep;  // still in the running for a local maximum?
};

// TODO : see if these hacks really are faster or not
#define fast_max(x,y) (x - ((x - y) & ((x - y) >> (sizeof(int) * CHAR_BIT - 1))))
#define fast_min(x,y) (y + ((x - y) & ((x - y) >> (sizeof(int) * CHAR_BIT - 1))))

void 
fast_nms(std::vector<DRect> &rects, float overlap_th, std::vector<DRect> &pick) 
{
   pick.clear(); // Clear out return array

   // Sort input rects by decreasing score - i.e. look at best
   // values first
   std::sort(rects.begin(), rects.end(), std::greater<DRect>());

   // Loop while there's anything valid left in rects array
   bool anyValid = true;
   do
   {
      anyValid = false; // assume there's nothing valid, adjust later if needed

      // Look for first valid entry in rects
      std::vector<DRect>::iterator it = rects.begin();
      for (; it != rects.end(); ++it)
	 if (it->valid())
	    break;

      // Exit if none are found
      if (it == rects.end())
	 break;

      // Save the highest ranked remaining DRect
      // and invalidate it - this means we've already
      // processed it
      pick.push_back(*it);
      it->invalidate();

      // Save coords of this DRect so we can
      // filter out nearby DRects which have a lower
      // ranking
      int x0 = it->x0();
      int y0 = it->y0();
      int x1 = it->x1();
      int y1 = it->y1();


      // Loop through the rest of the array, looking
      // for entries which overlap with the current "good"
      // one being processed
      for (++it; it != rects.end(); ++it) 
      {
	 if (it->valid())
	 {
	    int tx0 = fast_max(x0, it->x0());
	    int ty0 = fast_max(y0, it->y0());
	    int tx1 = fast_min(x1, it->x1());
	    int ty1 = fast_min(y1, it->y1());

	    tx0 = tx1 - tx0 + 1;
	    ty0 = ty1 - ty0 + 1;
	    if ((tx0 > 0) && (ty0 > 0) && ((tx0 * ty0 * it->invArea()) > overlap_th)) 
	       it->invalidate(); // invalidate DRects which overlap
	    else
	       anyValid = true;  // otherwise indicate that there's stuff left to do next time
	 }
      }
   }
   while (anyValid);
}

static void 
test_nn() 
{
   std::vector<DRect> rects;
   std::vector<DRect> keep;

   rects.push_back(DRect(0,  0,  10, 10, 0.5f));
   rects.push_back(DRect(1,  1,  10, 10, 0.4f));
   rects.push_back(DRect(20, 20, 40, 40, 0.3f));
   rects.push_back(DRect(20, 20, 40, 30, 0.4f));
   rects.push_back(DRect(15, 20, 40, 40, 0.1f));
   
   fast_nms(rects, 0.4f, keep);

   for (size_t i = 0; i < keep.size(); i++)
      keep[i].print();
}

int 
main(int argc, char *argv[]) 
{
    test_nn();
    return 0;
}

