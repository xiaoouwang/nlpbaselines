        self.ds_encoded.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )