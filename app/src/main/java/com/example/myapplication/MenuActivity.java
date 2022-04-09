package com.example.myapplication;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;
import androidx.navigation.NavController;

/*
This page shows the main three buttons in the application which are Rate Speech,
Voice Recorder, and Past Speech
 */
public class MenuActivity extends AppCompatActivity {
    private NavController navController;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_menu);

    }
    public void btnRateSpeech(View view) {
        //Calls the rating speech page
        Intent i = new Intent(this, InputSpeech.class);
        startActivity(i);
    }

    public void recorderPage(View view) {
        //Calls the voice recorder
        Intent i = new Intent(this,Recorder.class);
        startActivity(i);
    }

    public void btnPastSpeech(View view) {
        //Calls the past speeches pages to show speeches
        Intent intent= new Intent(MenuActivity.this, HistoryActivity.class);
        startActivity(intent);

    }
}