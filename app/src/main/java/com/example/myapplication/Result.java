package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import java.util.ArrayList;
import java.util.Collections;

public class Result extends AppCompatActivity {

    TextView text1;
    TextView text2;
    TextView text3;
    TextView text4;
    Button save;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);


        ArrayList<Integer> points_1 = new ArrayList<>();
        points_1.add(55);
        points_1.add(94);
        points_1.add(82);
        points_1.add(68);

        String [] recording_names = new String[4];
        recording_names[0]="Recording 1";
        recording_names[1]="Recording 2";
        recording_names[2]="Recording 3";
        recording_names[3]="Recording 4";

        ArrayList<Integer> points = new ArrayList<>();
        points.add(55);
        points.add(94);
        points.add(82);
        points.add(68);


        text1= findViewById(R.id.textView1);
        text2= findViewById(R.id.textView2);
        text3= findViewById(R.id.textView3);
        text4= findViewById(R.id.textView4);

        save = findViewById(R.id.save);

        Collections.sort(points,Collections.reverseOrder());
//        text1.setText(recording_names[0].toString());


        for (int i = 0; i <= 3; i++){
            if(points.get(0).equals(points_1.get(i))){
                text1.setText(recording_names[i] +"         "+ points_1.get(i).toString());
            }
        }

        for (int i = 0; i <= 3; i++){
            if(points.get(1).equals(points_1.get(i))){
                text2.setText(recording_names[i] +"         "+ points_1.get(i).toString());
            }
        }

        for (int i = 0; i <= 3; i++){
            if(points.get(2).equals(points_1.get(i))){
                text3.setText(recording_names[i] +"         "+ points_1.get(i).toString());
            }
        }

        for (int i = 0; i <= 3; i++) {
            if (points.get(3).equals(points_1.get(i))) {
                text4.setText(recording_names[i] + "         " + points_1.get(i).toString());

            }
        }
    }
}