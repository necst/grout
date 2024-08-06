package com.necst.controller.utils;

import org.graalvm.options.OptionKey;

public class OptionValuesMockBuilder {
    private final OptionValuesMock options;

    public OptionValuesMockBuilder() {
        this.options = new OptionValuesMock();
    }

    public <T> OptionValuesMockBuilder add(OptionKey<T> optionKey, T value) {
        this.options.set(optionKey, value);
        return this;
    }

    public OptionValuesMock build() {
        return options;
    }
}