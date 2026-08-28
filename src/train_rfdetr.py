from rfdetr import RFDETRSmall

if __name__ == "__main__":
    m = RFDETRSmall()
    m.train(dataset_dir="data/detector_coco", epochs=20, batch_size=4,
            grad_accum_steps=4, lr=1e-4, output_dir="runs/rfdetr", device="mps")
    print("RFDETR_DONE")
