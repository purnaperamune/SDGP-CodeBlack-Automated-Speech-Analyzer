package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

public class MainActivity3 extends AppCompatActivity {
    TextView recording_1;
    EditText input_1;
    TextView recording_2;
    EditText input_2;
    TextView recording_3;
    EditText input_3;
    TextView recording_4;
    EditText input_4;
    Button Enter;
    String in_1 = "";
    String in_2 = "";
    String in_3 = "";
    String in_4 = "";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main3);


        recording_1 = findViewById(R.id.recording_1);
        input_1 = findViewById(R.id.input_1);
        recording_2 = findViewById(R.id.recording_2);
        input_2 = findViewById(R.id.input_2);
        recording_3 = findViewById(R.id.recording_3);
        input_3 = findViewById(R.id.input_3);
        recording_4 = findViewById(R.id.recording_4);
        input_4 = findViewById(R.id.input_4);
        Enter = findViewById(R.id.Enter);
        Intent intent = getIntent();

        String path = intent.getStringExtra("message_key");
        recording_1.setText(path);
        String path1 = intent.getStringExtra("message_key1");
        recording_2.setText(path1);
        String path2 = intent.getStringExtra("message_key2");
        recording_3.setText(path2);
        String path3 = intent.getStringExtra("message_key3");
        recording_4.setText(path3);

        in_1 = input_1.getText().toString();
        in_2 = input_2.getText().toString();
        in_3 = input_3.getText().toString();
        in_4 = input_4.getText().toString();

        Enter.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity3.this,MainActivity2.class);
                startActivity(intent);

            }
        });
    }
}