/*
 * RuntimeBranching.h
 *
 */

#ifndef GC_RUNTIMEBRANCHING_H_
#define GC_RUNTIMEBRANCHING_H_

namespace GC
{

class RuntimeBranching
{
    bool tainted;

public:
	RuntimeBranching() : tainted(true)
	{
	}

    void untaint()
    {
        bool was_tainted = tainted;
        tainted = false;
        if (was_tainted)
            throw needs_cleaning();
        else
            taint();
    }
    void taint()
    {
        tainted = true;
    }

    bool is_tainted()
    {
        return tainted;
    }
};

} /* namespace GC */

#endif /* GC_RUNTIMEBRANCHING_H_ */
