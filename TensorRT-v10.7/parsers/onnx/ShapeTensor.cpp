/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ShapeTensor.hpp"
#include "Status.hpp"
#include "TensorOrWeights.hpp"
#include "importerUtils.hpp"
#include <algorithm>
#include <functional>

namespace onnx2trt
{

ShapeTensor::ShapeTensor(int32_t rank_, std::vector<int64_t>&& values_)
    : mDepth(0)
    , mAllValuesKnown(true)
    , mRank(rank_)
    , mSize(values_.size())
    , mValues(std::move(values_))
{
    assert((rank_ == 0 || rank_ == 1) && "shape tensor must have rank 0 or 1");
    assert(rank_ > 0 || mValues.size() == 1);
}

//!
//! Construct a shape tensor representation for float values.
//! This is a hack to get shape tensor working with float values with minimal changes.
//! The constructed ShapeTensor has the following properties:
//! 1. mAllValuesKnown == false
//! 2. mValues.size() == 0
//! 3. mIsFloat == true
//!
//! Float shape tensor does not have any constant folding in parser as available to int32_t shape tensors.
//! Instead, rely on builder for constant folding and build-time simplifications.
//!
ShapeTensor::ShapeTensor(int32_t rank_, std::vector<float>&& values_)
    : mDepth(0)
    , mAllValuesKnown(false)
    , mRank(rank_)
    , mSize(values_.size())
    , mValues({})
    , mIsFloat{true}
{
    assert((rank_ == 0 || rank_ == 1) && "shape tensor must have rank 0 or 1");
    assert(mValues.size() == 0 && "non-empty floating-point shape tensor with known values not supported");
}

ShapeTensor::ShapeTensor(ImporterContext* ctx, TensorOrWeights& t)
    : mDepth(0)
    , mIsFloat(t.isFp32())
{
    if (t.is_tensor())
    {
        *this = ShapeTensor(t.tensor());
    }
    else
    {
        const nvinfer1::Dims d = t.shape();
        auto const& weights = t.weights();
        if (isFloat())
        {
            *this = ShapeTensor(convertToTensor(t, ctx));
            return;
        }
        assert(0 <= d.nbDims);
        assert(d.nbDims <= 1 && "shape tensor must be 0D or 1D");
        mRank = d.nbDims;
        mSize = d.nbDims == 0 ? 1 : d.d[0];

        weightsToVector(weights, &mValues);
        mAllValuesKnown = true;
    }
}

static bool hasAllNonNegativeValues(const std::vector<int64_t>& values)
{
    return std::all_of(values.begin(), values.end(), [](int x) { return x >= 0; });
}

ShapeTensor::ShapeTensor(nvinfer1::ITensor& t, int depth)
    : mDepth(depth)
    , mRank(1)
    , mTensor(&t)
    // The check for depth == 0 is needed because when depth > 0, it means *this represents the shape of mTensor, and
    // shapes always have integral type.
    , mIsFloat(depth == 0 && t.getType() == nvinfer1::DataType::kFLOAT)
{
    const nvinfer1::Dims dims = t.getDimensions();
    assert((!isFloat() || mDepth == 0) && "floating-point shape tensor must have depth == 0");

    switch (mDepth)
    {
    case 0:
        assert(t.getType() == nvinfer1::DataType::kINT32 || t.getType() == nvinfer1::DataType::kINT64
            || t.getType() == nvinfer1::DataType::kFLOAT);
        mRank = dims.nbDims;
        if (mRank == 0)
        {
            mSize = 1;
        }
        else if (mRank == 1)
        {
            mSize = dims.d[0];
        }
        else
        {
            assert(mRank == -1);
        }
        break;

    case 1:
        if (dims.nbDims >= 0)
        {
            mSize = dims.nbDims;
            mValues.resize(dims.nbDims);
            std::copy_n(dims.d, dims.nbDims, mValues.begin());
            mAllValuesKnown = hasAllNonNegativeValues(mValues);
        }
        break;

    case 2:
        mSize = 1;
        if (dims.nbDims >= 0)
        {
            mValues = {dims.nbDims};
            mAllValuesKnown = hasAllNonNegativeValues(mValues);
        }
        break;

    case 3:
        // Applying IShapeLayer three times always yields a 1D vector containing 1.
        mDepth = 0;
        mSize = 1;
        mValues = {1};
        mAllValuesKnown = true;
        mTensor = nullptr;
        break;

    default:
        // Though depths greater than 3 could be handled the same as 3, they are
        // likely a sign of a problem.  Depths less than 0 make no sense.
        assert(0);
        break;
    }
}

ShapeTensor shapeVector(int64_t value)
{
    return ShapeTensor(1, std::vector<int64_t>({value}));
}

ShapeTensor shapeScalar(int64_t value)
{
    return ShapeTensor(0, std::vector<int64_t>({value}));
}

bool ShapeTensor::valueKnown(int k) const
{
    assert(0 <= k);
    assert(k < mSize);
    return allValuesKnown() || (mValues.size() == static_cast<size_t>(mSize) && mValues[k] >= 0);
}

bool ShapeTensor::isAll(int64_t x) const
{
    assert(mDepth >= 0 && "undefined tensor");
    return allValuesKnown() && std::all_of(begin(), end(), [x](int64_t y) { return x == y; });
}

nvinfer1::ITensor& ShapeTensor::tensor(ImporterContext* ctx) const
{
    assert(mDepth >= 0 && "undefined tensor");
    assert(mDepth <= 2);
    if (!mTensor || mDepth != 0)
    {
        // Need to create an ITensor representing *this.
        if (allValuesKnown())
        {
            // Create constant
            const nvinfer1::Dims dims{rank(), {size()}};
            // Must create temp weights in the parser in order to keep ownership of weights.
            auto shapeWeights = ctx->createNamedTempWeights(::ONNX_NAMESPACE::TensorProto::INT64, dims);
            std::memcpy(shapeWeights.values, mValues.data(), mValues.size() * sizeof(int64_t));
            auto* constLayer = N_CHECK(ctx->network()->addConstant(dims, shapeWeights));
            ctx->registerLayer(constLayer, "ONNXTRT_ShapeTensorFromDims", nullptr);
            mTensor = N_CHECK(constLayer->getOutput(0));
            mDepth = 0;
        }
        else
        {
            assert(mTensor);
            for (; mDepth > 0; --mDepth)
            {
                auto* shapeLayer = N_CHECK(ctx->network()->addShape(*mTensor));
                ctx->registerLayer(shapeLayer, "ONNXTRT_ShapeTensor", nullptr);
                mTensor = N_CHECK(shapeLayer->getOutput(0));
                mTensor = castHelper(ctx, mTensor, nvinfer1::DataType::kINT64);
            }
        }
    }
    return *mTensor;
}

ShapeTensor iotaShapeVector(int32_t n)
{
    std::vector<int64_t> values(n);
    std::iota(values.begin(), values.end(), 0);
    return ShapeTensor(1, std::move(values));
}

ShapeTensor similar(ImporterContext* ctx, const ShapeTensor& exemplar, int64_t value)
{
    return fillShapeVector(ctx, value, shapeOf(exemplar));
}

ShapeTensor fillShapeVector(ImporterContext* ctx, int64_t value, const ShapeTensor& count)
{
    assert(count.rank() == 1 && "implementation assumes 1D size");
    assert(count.size() == 1 && "implementation assumes 1D size of known size");
    if (count.allValuesKnown())
    {
        return ShapeTensor(1, std::vector<int64_t>(count[0], value));
    }
    else
    {
        nvinfer1::ISliceLayer* slice
            = addSlice(ctx, shapeVector(value).tensor(ctx), shapeVector(0), count, shapeVector(0));
        ctx->registerLayer(slice, "ONNXTRT_FillShapeVector", nullptr);
        auto* sliceOutput = N_CHECK(slice->getOutput(0));
        return ShapeTensor(*sliceOutput);
    }
}

using nvinfer1::ElementWiseOperation;

//! Helper that implements an elementwise operations on two shape tensors x and y.
//! f must implement the operation on a pair of int64_t.
//! commutes should be true f is commutative.
//! rightIdentity should be the right identity value for f.
static ShapeTensor op(ImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y, ElementWiseOperation operation,
    bool commutative, int64_t rightIdentity, const std::function<int64_t(int64_t, int64_t)>&& f)
{
    assert(!x.rankKnown() || !y.rankKnown() || x.rank() == y.rank());
    // Return early for empty-vector operands -- when present, the result is empty too.
    // Returning early for empty-vector operands simplifies subsequent logic, which
    // can assume the size of the result is the max of the sizes of the operands.
    if (x.isEmpty())
    {
        return x;
    }
    if (y.isEmpty())
    {
        return y;
    }
    if (x.sizeKnown() && y.sizeKnown())
    {
        assert(x.size() == 1 || y.size() == 1 || x.size() == y.size());
        if (y.isAll(rightIdentity) && y.size() <= x.size())
        {
            return x;
        }
        if (commutative && x.isAll(rightIdentity) && x.size() <= y.size())
        {
            return y;
        }
    }
    if (x.allValuesKnown() && y.allValuesKnown())
    {
        assert(!x.isFloat() && !y.isFloat());
        std::vector<int64_t> values(std::max(x.size(), y.size()));
        for (size_t i = 0; i < values.size(); ++i)
        {
            // The % simulates broadcast rules.
            values[i] = f(x[i % x.size()], y[i % y.size()]);
        }
        return ShapeTensor(x.rank(), std::move(values));
    }
    auto* elemLayer = N_CHECK(ctx->network()->addElementWise(x.tensor(ctx), y.tensor(ctx), operation));
    ctx->registerLayer(elemLayer, "ONNXTRT_ShapeElementWise", nullptr);
    auto* elemOutput = N_CHECK(elemLayer->getOutput(0));
    return ShapeTensor(*elemOutput, 0);
}

ShapeTensor add(ImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y)
{
    return op(ctx, x, y, ElementWiseOperation::kSUM, true, 0, std::plus<int64_t>());
}

ShapeTensor sub(ImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y)
{
    return op(ctx, x, y, ElementWiseOperation::kSUB, false, 0, std::minus<int64_t>());
}

ShapeTensor mul(ImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y)
{
    return op(ctx, x, y, ElementWiseOperation::kPROD, true, 1, std::multiplies<int64_t>());
}

ShapeTensor min(ImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y)
{
    return op(ctx, x, y, ElementWiseOperation::kMIN, true, std::numeric_limits<int64_t>::max(),
        [](int64_t x, int64_t y) { return std::min(x, y); });
}

ShapeTensor max(ImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y)
{
    return op(ctx, x, y, ElementWiseOperation::kMAX, true, std::numeric_limits<int64_t>::min(),
        [](int64_t x, int64_t y) { return std::max(x, y); });
}
ShapeTensor floorDiv(ImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y)
{
    return op(ctx, x, y, ElementWiseOperation::kFLOOR_DIV, false, 1, [](int64_t x, int64_t y) {
        assert(y != 0 && "divisor must be non-zero");
        const int64_t d = x / y;
        return d * y == x ? d : d - ((x < 0) ^ (y < 0));
    });
}

ShapeTensor broadcast(ImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y)
{
    // max(x,y) works unless x or y is 0.
    // min(x,y,1) yields 0 if x or y is 0, and 1 otherwise.
    // So compute max(x,y)*min(x,y,1).
    return mul(ctx, max(ctx, x, y), min(ctx, x, min(ctx, y, similar(ctx, y, 1))));
}

ShapeTensor product(ImporterContext* ctx, const ShapeTensor& x, int first, int last, int rank)
{
    assert(first <= last);
    ShapeTensor z(rank, std::vector<int64_t>(1, 1));
    for (int i = first; i < last; ++i)
    {
        z = mul(ctx, z, gather(ctx, x, ShapeTensor(rank, std::vector<int64_t>(1, i))));
    }
    return z;
}

ShapeTensor concat(ImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y)
{
    assert(!x.rankKnown() || x.rank() == 1);
    assert(!y.rankKnown() || y.rank() == 1);
    if (x.sizeKnown() && x.size() == 0)
    {
        return y;
    }
    if (y.sizeKnown() && y.size() == 0)
    {
        return x;
    }
    if (x.allValuesKnown() && y.allValuesKnown())
    {
        std::vector<int64_t> values(x.size() + y.size());
        auto p = std::copy(x.begin(), x.end(), values.begin());
        std::copy(y.begin(), y.end(), p);
        return ShapeTensor(1, std::move(values));
    }

    nvinfer1::ITensor* const args[2] = {&x.tensor(ctx), &y.tensor(ctx)};
    auto* concatLayer = N_CHECK(ctx->network()->addConcatenation(args, 2));
    ctx->registerLayer(concatLayer, "ONNXTRT_ShapeConcat", nullptr);
    auto* concatOutput = N_CHECK(concatLayer->getOutput(0));
    return ShapeTensor(*concatOutput);
}

ShapeTensor gather(ImporterContext* ctx, const ShapeTensor& data, const ShapeTensor& indices)
{
    assert(data.rank() == 1);
    if (indices.allValuesKnown()
        && std::all_of(indices.begin(), indices.end(), [&data](int i) { return data.valueKnown(i); }))
    {
        std::vector<int64_t> z(indices.size());
        std::transform(indices.begin(), indices.end(), z.begin(), [&data](int64_t i) {
            assert(0 <= i);
            assert(i < data.size());
            return data[i];
        });
        return ShapeTensor(indices.rank(), std::move(z));
    }
    auto* gatherLayer = ctx->network()->addGather(data.tensor(ctx), indices.tensor(ctx), 0);
    ctx->registerLayer(gatherLayer, "ONNXTRT_ShapeGather", nullptr);
    auto* gatherOutput = N_CHECK(gatherLayer->getOutput(0));
    return ShapeTensor(*gatherOutput);
}

ShapeTensor castToInt32(ImporterContext* ctx, ShapeTensor const& x)
{
    nvinfer1::ILayer* cast = N_CHECK(ctx->network()->addCast(x.tensor(ctx), nvinfer1::DataType::kINT32));
    ctx->registerLayer(cast, "ONNXTRT_ShapeCastToInt32", nullptr);
    auto castOutput = N_CHECK(cast->getOutput(0));
    return ShapeTensor(*castOutput);
}

ShapeTensor castToInt64(ImporterContext* ctx, ShapeTensor const& x)
{
    nvinfer1::ILayer* cast = N_CHECK(ctx->network()->addCast(x.tensor(ctx), nvinfer1::DataType::kINT64));
    ctx->registerLayer(cast, "ONNXTRT_ShapeCastToInt64", nullptr);
    auto castOutput = N_CHECK(cast->getOutput(0));
    return ShapeTensor(*castOutput);
}

ShapeTensor shapeOf(nvinfer1::ITensor& tensor)
{
    return ShapeTensor(tensor, 1);
}

ShapeTensor shapeOf(TensorOrWeights& t)
{
    if (t.is_tensor())
    {
        return shapeOf(t.tensor());
    }
    const nvinfer1::Dims& d = t.weights().shape;
    return ShapeTensor(1, std::vector<int64_t>(d.d, d.d + d.nbDims));
}

ShapeTensor shapeOf(const ShapeTensor& t)
{
    assert(t.mDepth >= 0);
    if (t.mTensor)
    {
        return ShapeTensor(*t.mTensor, t.mDepth + 1);
    }
    assert(t.rankKnown());
    assert(t.sizeKnown());
    // ShapeTensor is either a scalar or vector.
    // shape of a scalar is an empty tensor.
    // shape of a vector is a one-element tensor containing the length of the vector.
    return t.rank() == 0 ? ShapeTensor(0, std::vector<int64_t>{}) : ShapeTensor(1, std::vector<int64_t>{t.size()});
}

ShapeTensor convertTo1D(ImporterContext* ctx, const ShapeTensor& tensor)
{
    assert(tensor.rank() == 0);
    assert(tensor.size() == 1);
    if (tensor.valueKnown(0))
    {
        return shapeVector(tensor[0]);
    }
    return ShapeTensor(*N_CHECK(addShuffle(ctx, tensor.tensor(ctx), shapeVector(1))->getOutput(0)));
}

ShapeTensor convertTo0D(ImporterContext* ctx, const ShapeTensor& tensor)
{
    if (tensor.size() != 1)
    {
        throw std::runtime_error("Cannot convert a tensor with size > 1 to a scalar!");
    }
    if (tensor.valueKnown(0))
    {
        return shapeScalar(tensor[0]);
    }
    auto* layer = N_CHECK(ctx->network()->addShuffle(tensor.tensor(ctx)));
    layer->setReshapeDimensions(nvinfer1::Dims{0});
    return ShapeTensor(*N_CHECK(layer->getOutput(0)));
}

//! If all values of x are known, return Dims with those values,
//! but throw exception if any value is outside specified bounds.
//! Otherwise return Dims with zeros.
//!
//! The string that should describe the context of the dimensions,
//! e.g. "reshape" or "fill output".
nvinfer1::Dims shapeTensorToDims(const ShapeTensor& x, const char* what, int32_t minAllowed, int32_t maxAllowed)
{
    nvinfer1::Dims d{-1, {}};
    if (x.sizeKnown())
    {
        d.nbDims = x.size();
        if (x.allValuesKnown())
        {
            assert(x.size() <= nvinfer1::Dims::MAX_DIMS);
            for (const auto& dim : x)
            {
                if (dim < minAllowed || dim > maxAllowed)
                {
                    std::ostringstream msg;
                    msg << what << " dimensions have value " << dim << " beyond allowed bounds." << std::endl;
                    throw std::runtime_error(msg.str());
                }
            }
            std::copy(x.begin(), x.end(), d.d);
        }
    }
    return d;
}

//! If not all values in x are known, set layer input specifed by inputIndex
//! to tensor with value of x.
static void setShapeInputIfDynamic(ImporterContext* ctx, nvinfer1::ILayer* layer, int inputIndex, const ShapeTensor& x)
{
    if (!x.allValuesKnown())
    {
        layer->setInput(inputIndex, x.tensor(ctx));
    }
}

bool operator==(const ShapeTensor& x, const ShapeTensor& y)
{
    if (x.allValuesKnown() && y.allValuesKnown())
    {
        return x.mValues == y.mValues;
    }
    assert(x.mTensor || y.mTensor);
    return x.mTensor == y.mTensor && x.mDepth == y.mDepth;
}

nvinfer1::ITensor& reshape(ImporterContext* ctx, nvinfer1::ITensor& data, const ShapeTensor& newShape)
{
    const ShapeTensor oldShape = shapeOf(data);
    if (newShape == oldShape)
    {
        return data;
    }
    return *N_CHECK(addShuffle(ctx, data, newShape)->getOutput(0));
}

nvinfer1::IShuffleLayer* addShuffle(
    ImporterContext* ctx, nvinfer1::ITensor& data, const ShapeTensor& reshapeDims, bool zeroIsPlaceholder)
{
    nvinfer1::IShuffleLayer* shuffle = N_CHECK(ctx->network()->addShuffle(data));
    if (reshapeDims.allValuesKnown())
    {
        shuffle->setReshapeDimensions(
            shapeTensorToDims(reshapeDims, "reshape", -1, std::numeric_limits<int32_t>::max()));
    }
    else
    {
        shuffle->setInput(1, reshapeDims.tensor(ctx));
    }
    shuffle->setZeroIsPlaceholder(zeroIsPlaceholder);
    ctx->registerLayer(shuffle, "ONNXTRT_ShapeShuffle", nullptr);
    return shuffle;
}

nvinfer1::ISliceLayer* addSlice(ImporterContext* ctx, nvinfer1::ITensor& data, const ShapeTensor& starts,
    const ShapeTensor& sizes, const ShapeTensor& strides)
{
    constexpr int32_t minDim = std::numeric_limits<int32_t>::min();
    constexpr int32_t maxDim = std::numeric_limits<int32_t>::max();
    nvinfer1::ISliceLayer* slice = N_CHECK(ctx->network()->addSlice(data,
        shapeTensorToDims(starts, "slice start", minDim, maxDim), shapeTensorToDims(sizes, "slice size", 0, maxDim),
        shapeTensorToDims(strides, "slide strides", minDim, maxDim)));
    setShapeInputIfDynamic(ctx, slice, 1, starts);
    setShapeInputIfDynamic(ctx, slice, 2, sizes);
    setShapeInputIfDynamic(ctx, slice, 3, strides);
    ctx->registerLayer(slice, "ONNXTRT_ShapeSlice", nullptr);
    return slice;
}

nvinfer1::IFillLayer* addFill(ImporterContext* ctx, const ShapeTensor& shape, nvinfer1::FillOperation op)
{
    nvinfer1::IFillLayer* fill = N_CHECK(
        ctx->network()->addFill(shapeTensorToDims(shape, "fill output", 0, std::numeric_limits<int32_t>::max()), op,
            nvinfer1::DataType::kFLOAT));
    setShapeInputIfDynamic(ctx, fill, 0, shape);
    ctx->registerLayer(fill, "ONNXTRT_ShapeFill", nullptr);
    return fill;
}

std::ostream& operator<<(std::ostream& stream, const ShapeTensor& x)
{
    stream << "(";
    for (int i = 0, e = x.size(); i < e; ++i)
    {
        stream << (i ? ", " : "");
        if (x.valueKnown(i))
        {
            stream << x[i];
        }
        else
        {
            stream << "_";
        }
    }
    if (x.size() == 1 && x.rank() == 1)
    {
        // Use Python convention to distinguish 1-element vector from a scalar.
        stream << ",";
    }
    return stream << ")";
}

} // namespace onnx2trt
