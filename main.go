package main

import (
	"fmt"
	"time"

	"github.com/unixpickle/cuda"
	"github.com/unixpickle/cuda/cublas"
	"github.com/unixpickle/cuda/curand"
	"github.com/unixpickle/essentials"
)

const MatrixSize = 1 << 10

func main() {
	fmt.Println("Finding device...")
	devices, err := cuda.AllDevices()
	if err != nil {
		essentials.Die(err)
	}
	if len(devices) == 0 {
		essentials.Die("no device found")
	}
	ctx, err := cuda.NewContext(devices[0], 10)
	if err != nil {
		essentials.Die(err)
	}

	err = <-ctx.Run(func() error {
		blasHandle, err := cublas.NewHandle(ctx)
		if err != nil {
			return err
		}

		fmt.Println("Allocating matrices...")
		allocator := cuda.GCAllocator(cuda.NativeAllocator(ctx), 0)
		var bufs []cuda.Buffer
		for i := 0; i < 3; i++ {
			buf, err := cuda.AllocBuffer(allocator, MatrixSize*MatrixSize*4)
			if err != nil {
				return err
			}
			bufs = append(bufs, buf)
		}

		fmt.Println("Randomizing matrices...")
		gen, err := curand.NewGenerator(ctx, curand.PseudoDefault)
		if err != nil {
			return err
		}
		for _, buf := range bufs {
			if err := gen.Normal(buf, 0, 1); err != nil {
				return err
			}
		}

		fmt.Println("Timing matrix multiplications...")
		for {
			start := time.Now().UnixNano()
			for i := 0; i < 10; i++ {
				err := blasHandle.Sgemm(cublas.NoTrans, cublas.NoTrans, MatrixSize,
					MatrixSize, MatrixSize, float32(2.718), bufs[0], MatrixSize,
					bufs[1], MatrixSize, float32(3.142), bufs[2], MatrixSize)
				cuda.Synchronize()
				if err != nil {
					return err
				}
			}
			fmt.Println("Took", time.Now().UnixNano()-start, "ns")
			time.Sleep(time.Second)
		}
	})
	if err != nil {
		essentials.Die(err)
	}
}
